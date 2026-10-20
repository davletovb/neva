//! Asynchronous, bounded effect boundary with keyed record/replay and
//! fsynced effect receipts. World reducers never perform model/network I/O.
use crate::core::AgentId;
use crate::effects::{EffectCallKey, EffectResult, ModelRequest, Sampling, Usage};
use crate::storage::{MAX_EVENT_RECORD_BYTES, digest, read_record_limited, stable_bytes};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    future::Future,
    io::{BufReader, Write},
    path::{Path, PathBuf},
    pin::Pin,
    sync::Arc,
    time::Duration,
};
use tokio::{sync::Semaphore, task::JoinSet, time::timeout};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum EffectSpec {
    Model(ModelRequest),
    Tool {
        name: String,
        input: serde_json::Value,
    },
}

impl EffectSpec {
    pub fn key(
        &self,
        trial: &str,
        step: u64,
        agent: AgentId,
        index: u32,
    ) -> Result<EffectCallKey, String> {
        Ok(EffectCallKey {
            trial_id: trial.into(),
            logical_step: step,
            agent_id: agent,
            call_index: index,
            canonical_request_hash: digest(self)?,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BillingDisposition {
    NotCharged,
    Charged,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Receipt {
    pub key: EffectCallKey,
    pub spec: EffectSpec,
    pub result: EffectResult,
    /// An HTTP error or timeout can occur after the provider charges.
    pub billing: BillingDisposition,
    pub effective_provider: String,
    pub effective_model: Option<String>,
    /// Detects tampered model output/usage/identity during strict replay.
    pub content_hash: String,
}
impl Receipt {
    fn compute_hash(&self) -> Result<String, String> {
        digest(&(
            &self.key,
            &self.spec,
            &self.result,
            &self.billing,
            &self.effective_provider,
            &self.effective_model,
        ))
    }
}

pub type BoxResult<'a> = Pin<Box<dyn Future<Output = Result<EffectResult, String>> + Send + 'a>>;
pub trait Backend: Send + Sync {
    /// Called *after* the timed-out future has been cancelled. The provider
    /// may have processed/billed the request, so this only updates breaker
    /// state; it never retries or refunds the uncertain transaction.
    fn on_timeout<'a>(
        &'a self,
        _spec: &'a EffectSpec,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        Box::pin(async {})
    }
    /// Explicit guarantee: UnsupportedCapability never follows a dispatch.
    /// Default custom backends are not trusted to make this assertion.
    fn unsupported_is_pre_dispatch(&self) -> bool {
        false
    }
    /// Opt-in only: tool execution requires a backend with explicit guards.
    fn permits_tools(&self) -> bool {
        false
    }
    fn resolve_keyed<'a>(
        &'a self,
        _key: &'a EffectCallKey,
        request: &'a EffectSpec,
    ) -> BoxResult<'a> {
        self.resolve(request)
    }
    fn billing_disposition(
        &self,
        spec: &EffectSpec,
        response: &EffectResult,
    ) -> BillingDisposition {
        if matches!(response, EffectResult::UnsupportedCapability { .. }) {
            if self.unsupported_is_pre_dispatch() {
                BillingDisposition::NotCharged
            } else {
                BillingDisposition::Unknown
            }
        } else if matches!(spec, EffectSpec::Tool { .. }) {
            BillingDisposition::Unknown
        } else if matches!(response, EffectResult::Model { .. }) {
            BillingDisposition::Charged
        } else {
            BillingDisposition::Unknown
        }
    }
    fn effective_provider(&self, spec: &EffectSpec) -> String {
        match spec {
            EffectSpec::Tool { name, .. } => format!("tool:{name}"),
            _ => self.backend_id().into(),
        }
    }
    /// Useful for distinguishing scripted from network-backed billing semantics.
    fn backend_id(&self) -> &'static str {
        "custom"
    }
    fn resolve<'a>(&'a self, request: &'a EffectSpec) -> BoxResult<'a>;
}

/// Deterministic offline backend. Tools have no implicit executor.
#[derive(Default)]
pub struct ScriptedBackend;
impl Backend for ScriptedBackend {
    fn billing_disposition(
        &self,
        spec: &EffectSpec,
        response: &EffectResult,
    ) -> BillingDisposition {
        match spec {
            EffectSpec::Model(_) => BillingDisposition::NotCharged,
            _ => {
                if matches!(response, EffectResult::UnsupportedCapability { .. }) {
                    BillingDisposition::NotCharged
                } else {
                    BillingDisposition::Unknown
                }
            }
        }
    }
    fn unsupported_is_pre_dispatch(&self) -> bool {
        true
    }
    fn backend_id(&self) -> &'static str {
        "scripted"
    }
    fn resolve<'a>(&'a self, spec: &'a EffectSpec) -> BoxResult<'a> {
        Box::pin(async move {
            match spec {
                EffectSpec::Model(model) => {
                    let text = model
                        .messages
                        .iter()
                        .filter(|m| matches!(m.role, crate::effects::Role::User))
                        .map(|m| m.content.as_str())
                        .collect::<Vec<_>>()
                        .join("\n");
                    Ok(EffectResult::Model {
                        content: format!("scripted:{text}"),
                        effective_model: model.model.clone(),
                        usage: Some(Usage {
                            input_tokens: None,
                            output_tokens: None,
                            billed_cost: Some("0".into()),
                            currency: Some("USD".into()),
                        }),
                    })
                }
                EffectSpec::Tool { name, .. } => Ok(EffectResult::UnsupportedCapability {
                    capability: format!("no scripted implementation for tool {name}"),
                }),
            }
        })
    }
}

/// Replay from a validated receipt map by *full logical key* and request content.
/// Never assume completion-order cursors.
#[derive(Clone, Default)]
pub struct ReplayBackend {
    pub captured: BTreeMap<EffectCallKey, Receipt>,
}
impl ReplayBackend {
    pub fn from_receipts(receipts: impl IntoIterator<Item = Receipt>) -> Self {
        Self {
            captured: receipts.into_iter().map(|r| (r.key.clone(), r)).collect(),
        }
    }
    pub fn resolve_key(&self, key: &EffectCallKey, spec: &EffectSpec) -> Result<Receipt, String> {
        let existing = self
            .captured
            .get(key)
            .ok_or("missing keyed replay effect")?;
        if existing.content_hash != existing.compute_hash()?
            || &existing.spec != spec
            || spec.key(
                &key.trial_id,
                key.logical_step,
                key.agent_id,
                key.call_index,
            )? != *key
        {
            return Err("keyed replay request mismatch".into());
        }
        Ok(existing.clone())
    }
}

/// Single-writer file lease for standalone effect journals.
struct EffectWriterLock {
    path: PathBuf,
    file: Option<std::fs::File>,
}
impl Drop for EffectWriterLock {
    fn drop(&mut self) {
        if let Some(file) = self.file.take() {
            let _ = file.sync_all();
            drop(file);
        }
        let _ = fs::remove_file(&self.path);
    }
}

pub struct EffectJournal {
    pub path: PathBuf,
    _lock: EffectWriterLock,
    pub(crate) receipts: BTreeMap<EffectCallKey, Receipt>,
    poisoned: bool,
    owning_trial: Option<String>,
}

/// Per-record cap for receipts (same bound as durable events). Total file
/// bound also caps aggregate replay indexing and avoids unbounded history.
pub const MAX_EFFECT_JOURNAL_BYTES: u64 = 256 * 1024 * 1024;

/// Iterate independently verified lines; writer mode tolerates only a final
/// unfinished fragment and may truncate it under its exclusive writer lock.
fn scan_receipts(
    path: &Path,
    allow_torn_tail: bool,
    mut on_receipt: impl FnMut(Receipt) -> Result<(), String>,
) -> Result<u64, String> {
    let input = File::open(path).map_err(|e| e.to_string())?;
    let mut reader = BufReader::new(input);
    let mut offset = 0u64;
    let mut line = Vec::new();
    loop {
        let count = read_record_limited(&mut reader, &mut line)?;
        if count == 0 {
            break;
        }
        let next = offset
            .checked_add(count as u64)
            .ok_or("effect ledger offset overflow")?;
        if next > MAX_EFFECT_JOURNAL_BYTES {
            return Err("effect journal exceeds portable total size limit".into());
        }
        if !line.ends_with(b"\n") {
            if allow_torn_tail {
                break;
            }
            return Err("effect log has uncommitted/truncated final record".into());
        }
        if line == b"\n" {
            return Err("blank committed receipt record".into());
        }
        let receipt: Receipt =
            serde_json::from_slice(&line).map_err(|e| format!("invalid effect ledger: {e}"))?;
        if receipt.content_hash != receipt.compute_hash()?
            || receipt.spec.key(
                &receipt.key.trial_id,
                receipt.key.logical_step,
                receipt.key.agent_id,
                receipt.key.call_index,
            )? != receipt.key
        {
            return Err("effect ledger receipt hash/request identity mismatch".into());
        }
        on_receipt(receipt)?;
        offset = next;
    }
    Ok(offset)
}

impl EffectJournal {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, String> {
        let lock_path = dir.as_ref().join(".effects-writer.lock");
        let lock_file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&lock_path)
            .map_err(|e| format!("effect journal already open or stale lock: {e}"))?;
        let lock = EffectWriterLock {
            path: lock_path,
            file: Some(lock_file),
        };
        let path = dir.as_ref().join("effects.jsonl");
        if !path.exists() {
            OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&path)
                .and_then(|f| f.sync_all())
                .map_err(|e| e.to_string())?;
            crate::storage::sync_trial_directory(dir.as_ref())?;
        }
        let mut receipts = BTreeMap::new();
        let length = scan_receipts(&path, true, |receipt| {
            if receipts.insert(receipt.key.clone(), receipt).is_some() {
                return Err("duplicate effect ledger key".into());
            }
            Ok(())
        })?;
        let physical = fs::metadata(&path).map_err(|e| e.to_string())?.len();
        if physical > MAX_EFFECT_JOURNAL_BYTES {
            return Err("effect journal exceeds portable total size limit".into());
        }
        if length < physical {
            let f = OpenOptions::new()
                .write(true)
                .open(&path)
                .map_err(|e| e.to_string())?;
            f.set_len(length)
                .and_then(|_| f.sync_all())
                .map_err(|e| e.to_string())?;
        }
        Ok(Self {
            path,
            _lock: lock,
            receipts,
            poisoned: false,
            owning_trial: None,
        })
    }

    /// Bind both existing and future receipts to the committed trial genesis.
    pub fn bind_trial(&mut self, trial: &str) -> Result<(), String> {
        if self.receipts.keys().any(|key| key.trial_id != trial) {
            return Err("foreign effect receipt found for this trial".into());
        }
        self.owning_trial = Some(trial.to_owned());
        Ok(())
    }

    /// Read-only ledger validation: no writer lock, creation or truncation.
    pub fn inspect_read_only(dir: impl AsRef<Path>, trial: &str) -> Result<usize, String> {
        let mut seen = std::collections::BTreeSet::new();
        scan_receipts(&dir.as_ref().join("effects.jsonl"), false, |receipt| {
            if receipt.key.trial_id != trial || !seen.insert(receipt.key.clone()) {
                return Err("effect ledger failed read-only trial/integrity check".into());
            }
            Ok(())
        })?;
        Ok(seen.len())
    }

    pub fn open_existing(dir: impl AsRef<Path>) -> Result<Self, String> {
        if !dir.as_ref().join("effects.jsonl").exists() {
            return Err("missing effect ledger".into());
        }
        Self::open(dir)
    }

    pub fn receipt_count(&self) -> usize {
        self.receipts.len()
    }
    pub fn receipts(&self) -> &BTreeMap<EffectCallKey, Receipt> {
        &self.receipts
    }

    fn append(&mut self, mut receipt: Receipt) -> Result<Receipt, String> {
        if self.poisoned {
            return Err("effect ledger poisoned, reopen to recover".into());
        }
        if self.receipts.contains_key(&receipt.key) {
            return Err("effect already recorded".into());
        }
        if self
            .owning_trial
            .as_ref()
            .is_some_and(|trial| trial != &receipt.key.trial_id)
        {
            return Err("attempt to write effect receipt for another trial".into());
        }
        receipt.content_hash = receipt.compute_hash()?;
        let mut bytes = stable_bytes(&receipt)?;
        bytes.push(b'\n');
        if bytes.len() > MAX_EVENT_RECORD_BYTES {
            return Err("effect receipt exceeds portable record size limit".into());
        }
        let total = fs::metadata(&self.path).map_err(|e| e.to_string())?.len();
        if total
            .checked_add(bytes.len() as u64)
            .is_none_or(|n| n > MAX_EFFECT_JOURNAL_BYTES)
        {
            return Err("effect journal exceeds portable total size limit".into());
        }
        if let Err(e) = (|| -> std::io::Result<()> {
            let mut f = OpenOptions::new().append(true).open(&self.path)?;
            f.write_all(&bytes)?;
            f.sync_all()
        })() {
            self.poisoned = true;
            return Err(format!("effect receipt fsync failure: {e}"));
        }
        self.receipts.insert(receipt.key.clone(), receipt.clone());
        Ok(receipt)
    }

    /// Concurrent model/tool requests; receipts are durable before returning to
    /// the deterministic reducer. Output order follows logical call keys.
    pub async fn resolve_all(
        &mut self,
        specs: Vec<(EffectCallKey, EffectSpec)>,
        backend: Arc<dyn Backend>,
        concurrency: usize,
        deadline: Duration,
        replay_only: bool,
    ) -> Result<Vec<Receipt>, String> {
        if concurrency == 0 {
            return Err("concurrency limit must be positive".into());
        }
        // Complete validation comes BEFORE launching any billable request.
        let mut seen = std::collections::BTreeSet::new();
        let mut pending = Vec::new();
        let mut finished = Vec::new();
        for (key, spec) in specs {
            if !seen.insert(key.clone()) {
                return Err("duplicate effect call key".into());
            }
            if self
                .owning_trial
                .as_ref()
                .is_some_and(|trial| trial != &key.trial_id)
                || key
                    != spec.key(
                        &key.trial_id,
                        key.logical_step,
                        key.agent_id,
                        key.call_index,
                    )?
            {
                return Err("effect trial/request key mismatch".into());
            }
            if let Some(existing) = self.receipts.get(&key) {
                if existing.spec != spec || existing.content_hash != existing.compute_hash()? {
                    return Err("recorded request content/hash mismatch".into());
                }
                finished.push(existing.clone());
            } else if replay_only {
                return Err(format!(
                    "missing recorded effect: {}",
                    key.canonical_request_hash
                ));
            } else {
                pending.push((key, spec));
            }
        }
        // Historic receipts with >32 calls remain valid to replay. Only
        // pending (newly billable) effects are subject to execution limits.
        if pending.len() > 32
            || pending
                .iter()
                .filter(|(_, spec)| matches!(spec, EffectSpec::Tool { .. }))
                .count()
                > 8
        {
            return Err("new effect batch exceeds limit (32 effects / 8 tools)".into());
        }
        // Replayed/previously recorded tool effects are inert and need no
        // live executor permission. Only validate pending executions.
        if pending
            .iter()
            .any(|(_, spec)| matches!(spec, EffectSpec::Tool { .. }))
            && !backend.permits_tools()
        {
            return Err("tool backend lacks explicit permissions/approvals".into());
        }
        if self.poisoned && !pending.is_empty() {
            return Err("effect ledger poisoned; reopen/recover before generating".into());
        }
        let sem = Arc::new(Semaphore::new(concurrency));
        let mut jobs = JoinSet::new();
        for (key, spec) in pending {
            let sem = sem.clone();
            let backend = backend.clone();
            jobs.spawn(async move {
                let _permit = sem.acquire_owned().await.map_err(|e| e.to_string())?;
                let result = timeout(deadline, backend.resolve_keyed(&key, &spec)).await;
                let response = match result {
                    Ok(Ok(value)) => value,
                    Ok(Err(error)) => EffectResult::Failed {
                        category: "backend".into(),
                        message: error,
                    },
                    Err(_) => {
                        backend.on_timeout(&spec).await;
                        EffectResult::Failed {
                            category: "timeout".into(),
                            message: "effect deadline exceeded".into(),
                        }
                    }
                };
                let model = match &response {
                    EffectResult::Model {
                        effective_model, ..
                    } => Some(effective_model.clone()),
                    _ => None,
                };
                let billing = backend.billing_disposition(&spec, &response);
                let provider = backend.effective_provider(&spec);
                Ok::<_, String>(Receipt {
                    key,
                    spec,
                    result: response,
                    billing,
                    effective_provider: provider,
                    effective_model: model,
                    content_hash: String::new(),
                })
            });
        }
        // Always join all requests, even if one receipt append fails.
        // No early return can cancel another already-sent provider call.
        let mut failures = Vec::new();
        while let Some(result) = jobs.join_next().await {
            match result {
                Ok(Ok(receipt)) => match self.append(receipt) {
                    Ok(recorded) => finished.push(recorded),
                    Err(error) => failures.push(error),
                },
                Ok(Err(error)) => failures.push(format!("effect task failed: {error}")),
                Err(error) => failures.push(format!("effect task panicked/cancelled: {error}")),
            }
        }
        if !failures.is_empty() {
            return Err(format!(
                "effect batch incomplete; reconcile provider charges before retry: {}",
                failures.join("; ")
            ));
        }
        finished.sort_by(|a, b| a.key.cmp(&b.key));
        Ok(finished)
    }
}

/// Opt-in OpenAI-compatible HTTP adapter. Network requests are never D1.
/// Unknown billing after timeout/network error is retained in the journal.
#[cfg(feature = "live-http")]
#[derive(Clone, Debug)]
pub struct HttpReliability {
    pub max_response_bytes: usize,
    /// Only an explicit HTTP 429 is retried; transport/timeouts and 5xx may
    /// already have incurred provider charges and are never auto-retried.
    pub retry_429: u8,
    /// Not universally supported by OpenAI-compatible servers; opt in.
    pub include_stream_usage: bool,
    pub retry_delay: Duration,
    pub circuit_threshold: u32,
    pub circuit_cooldown: Duration,
}
#[cfg(feature = "live-http")]
impl Default for HttpReliability {
    fn default() -> Self {
        Self {
            max_response_bytes: 1024 * 1024,
            retry_429: 0,
            include_stream_usage: false,
            retry_delay: Duration::from_millis(100),
            circuit_threshold: 3,
            circuit_cooldown: Duration::from_secs(30),
        }
    }
}
#[cfg(feature = "live-http")]
#[derive(Default)]
struct CircuitState {
    failures: u32,
    open_until: Option<tokio::time::Instant>,
}
#[cfg(feature = "live-http")]
pub struct OpenAiCompatible {
    client: reqwest::Client,
    url: String,
    api_key: String,
    reliability: HttpReliability,
    circuit: tokio::sync::Mutex<CircuitState>,
}
#[cfg(feature = "live-http")]
impl OpenAiCompatible {
    pub fn new(url: String, api_key: String) -> Result<Self, String> {
        let parsed = reqwest::Url::parse(&url).map_err(|e| e.to_string())?;
        if !parsed.username().is_empty() || parsed.password().is_some() {
            return Err("userinfo in live URL is forbidden".into());
        }
        // Endpoint admission and proxy bypass use one shared test, including IPv6.
        let is_loopback = parsed.host_str().is_some_and(|host| {
            let bare = host
                .strip_prefix('[')
                .and_then(|v| v.strip_suffix(']'))
                .unwrap_or(host);
            bare == "localhost"
                || bare
                    .parse::<std::net::IpAddr>()
                    .is_ok_and(|address| address.is_loopback())
        });
        let allowed = parsed.scheme() == "https" || (parsed.scheme() == "http" && is_loopback);
        if !allowed {
            return Err("live model URL must be HTTPS or parsed loopback HTTP".into());
        }
        // Default environment HTTP_PROXY/HTTPS_PROXY discovery could tunnel
        // a localhost bearer request through an untrusted proxy, even though
        // the URL itself is parsed and validated. Force *direct* loopback.
        let mut builder = reqwest::Client::builder().redirect(reqwest::redirect::Policy::none());
        if is_loopback {
            builder = builder.no_proxy();
        }
        let client = builder.build().map_err(|e| e.to_string())?;
        Ok(Self {
            client,
            url: parsed.to_string(),
            api_key,
            reliability: HttpReliability::default(),
            circuit: tokio::sync::Mutex::new(CircuitState::default()),
        })
    }
    pub fn with_reliability(mut self, reliability: HttpReliability) -> Result<Self, String> {
        if reliability.max_response_bytes < 1024
            || reliability.max_response_bytes > 8 * 1024 * 1024
            || reliability.retry_429 > 2
            || reliability.circuit_threshold == 0
        {
            return Err("invalid HTTP reliability bounds".into());
        }
        self.reliability = reliability;
        Ok(self)
    }
    async fn failure(&self) {
        let mut state = self.circuit.lock().await;
        state.failures = state.failures.saturating_add(1);
        if state.failures >= self.reliability.circuit_threshold {
            state.open_until =
                Some(tokio::time::Instant::now() + self.reliability.circuit_cooldown);
        }
    }
    async fn successful(&self) {
        let mut state = self.circuit.lock().await;
        state.failures = 0;
        state.open_until = None;
    }
    async fn allowed(&self) -> bool {
        let mut state = self.circuit.lock().await;
        if let Some(end) = state.open_until {
            if tokio::time::Instant::now() < end {
                return false;
            }
            state.failures = 0;
            state.open_until = None;
        }
        true
    }
}

#[cfg(feature = "live-http")]
fn has_tool_calls(value: Option<&serde_json::Value>) -> bool {
    value.is_some_and(|v| match v {
        serde_json::Value::Null => false,
        serde_json::Value::Array(values) => !values.is_empty(),
        _ => true,
    })
}
#[cfg(feature = "live-http")]
fn parse_usage(value: &serde_json::Value) -> Option<Usage> {
    value.get("usage").filter(|v| !v.is_null()).map(|u| Usage {
        input_tokens: u.get("prompt_tokens").and_then(|v| v.as_u64()),
        output_tokens: u.get("completion_tokens").and_then(|v| v.as_u64()),
        billed_cost: None,
        currency: None,
    })
}
#[cfg(feature = "live-http")]
struct SseAccumulator {
    line: Vec<u8>,
    frame: Vec<u8>,
    text: String,
    effective_model: Option<String>,
    usage: Option<Usage>,
    done: bool,
    seen_bytes: usize,
    limit: usize,
}
#[cfg(feature = "live-http")]
impl SseAccumulator {
    fn new(limit: usize) -> Self {
        Self {
            line: vec![],
            frame: vec![],
            text: String::new(),
            effective_model: None,
            usage: None,
            done: false,
            seen_bytes: 0,
            limit,
        }
    }
    fn feed(&mut self, bytes: &[u8]) -> Result<(), String> {
        self.seen_bytes = self
            .seen_bytes
            .checked_add(bytes.len())
            .ok_or("stream response overflow")?;
        if self.seen_bytes > self.limit {
            return Err("stream response exceeds byte cap".into());
        }
        for &byte in bytes {
            if self.done {
                break;
            }
            if byte == b'\n' {
                let mut line = std::mem::take(&mut self.line);
                if line.last() == Some(&b'\r') {
                    line.pop();
                }
                if line.is_empty() {
                    if !self.frame.is_empty() {
                        let frame = std::mem::take(&mut self.frame);
                        self.parse_frame(&frame)?;
                    }
                } else if let Some(payload) = line.strip_prefix(b"data:") {
                    if !self.frame.is_empty() {
                        self.frame.push(b'\n');
                    }
                    self.frame
                        .extend_from_slice(payload.strip_prefix(b" ").unwrap_or(payload));
                }
            } else {
                self.line.push(byte);
                if self.line.len() > self.limit {
                    return Err("SSE line too long".into());
                }
            }
        }
        Ok(())
    }
    fn parse_frame(&mut self, frame: &[u8]) -> Result<(), String> {
        if frame == b"[DONE]" {
            self.done = true;
            return Ok(());
        }
        let payload: serde_json::Value =
            serde_json::from_slice(frame).map_err(|e| format!("invalid SSE JSON: {e}"))?;
        if payload.get("error").is_some_and(|v| !v.is_null()) {
            return Err("provider returned SSE error frame".into());
        }
        if let Some(model) = payload.get("model").and_then(|v| v.as_str()) {
            self.effective_model = Some(model.into());
        }
        if let Some(usage) = parse_usage(&payload) {
            self.usage = Some(usage);
        }
        if let Some(delta) = payload
            .pointer("/choices/0/delta/content")
            .and_then(|v| v.as_str())
        {
            self.text.push_str(delta);
            if self.text.len() > self.limit {
                return Err("stream output exceeds cap".into());
            }
        }
        if has_tool_calls(payload.pointer("/choices/0/delta/tool_calls")) {
            return Err("tool call streaming requires explicit tool executor".into());
        }
        Ok(())
    }
    fn finish(mut self, fallback: &str) -> Result<EffectResult, String> {
        if !self.line.is_empty() || !self.frame.is_empty() {
            return Err("truncated SSE frame".into());
        }
        if !self.done {
            return Err("stream ended without [DONE]".into());
        }
        Ok(EffectResult::Model {
            content: std::mem::take(&mut self.text),
            effective_model: self.effective_model.unwrap_or_else(|| fallback.into()),
            usage: self.usage,
        })
    }
}
#[cfg(feature = "live-http")]
async fn bounded_bytes(mut response: reqwest::Response, cap: usize) -> Result<Vec<u8>, String> {
    let mut raw = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|e| e.to_string())? {
        if raw.len().checked_add(chunk.len()).is_none_or(|n| n > cap) {
            return Err("HTTP response exceeds configured byte cap".into());
        }
        raw.extend_from_slice(&chunk);
    }
    Ok(raw)
}
#[cfg(feature = "live-http")]
impl Backend for OpenAiCompatible {
    fn on_timeout<'a>(
        &'a self,
        spec: &'a EffectSpec,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        Box::pin(async move {
            if matches!(spec, EffectSpec::Model(_)) {
                self.failure().await;
            }
        })
    }
    fn unsupported_is_pre_dispatch(&self) -> bool {
        true
    }
    fn backend_id(&self) -> &'static str {
        "openai-compatible"
    }
    fn resolve<'a>(&'a self, spec: &'a EffectSpec) -> BoxResult<'a> {
        Box::pin(async move {
            let model = match spec {
                EffectSpec::Model(m) => m,
                _ => {
                    return Ok(EffectResult::UnsupportedCapability {
                        capability: "HTTP tool operations require a separate permitted executor"
                            .into(),
                    });
                }
            };
            if !model.tool_schemas.is_empty() || model.response_format.is_some() {
                return Ok(EffectResult::UnsupportedCapability {
                    capability: "tool schemas/structured output unsupported by this adapter".into(),
                });
            }
            if model.model.trim().is_empty()
                || model
                    .sampling
                    .max_output_tokens
                    .is_none_or(|n| n == 0 || n > 8192)
            {
                return Ok(EffectResult::UnsupportedCapability {
                    capability: "model and bounded max_output_tokens (1..=8192) required".into(),
                });
            }
            let roles: Result<Vec<_>, String> = model
                .messages
                .iter()
                .map(|m| {
                    let role = match m.role {
                        crate::effects::Role::System => "system",
                        crate::effects::Role::User => "user",
                        crate::effects::Role::Assistant => "assistant",
                        crate::effects::Role::Tool => "tool",
                    };
                    let mut message = serde_json::json!({"role":role,"content":m.content});
                    if matches!(m.role, crate::effects::Role::Tool) {
                        let id = m
                            .tool_call_id
                            .as_ref()
                            .ok_or("tool role message requires tool_call_id")?;
                        message["tool_call_id"] = serde_json::json!(id);
                    }
                    Ok::<_, String>(message)
                })
                .collect::<Result<Vec<_>, _>>();
            let roles = match roles {
                Ok(roles) => roles,
                Err(error) => return Ok(EffectResult::UnsupportedCapability { capability: error }),
            };
            let mut body = serde_json::json!({
                "model":model.model,"messages":roles,"stream":model.streaming
            });
            if let Some(t) = model.sampling.temperature {
                body["temperature"] = serde_json::json!(t);
            }
            if let Some(p) = model.sampling.top_p {
                body["top_p"] = serde_json::json!(p);
            }
            if let Some(seed) = model.sampling.seed {
                body["seed"] = serde_json::json!(seed);
            }
            if let Some(max) = model.sampling.max_output_tokens {
                body["max_tokens"] = serde_json::json!(max);
            }
            if model.streaming && self.reliability.include_stream_usage {
                body["stream_options"] = serde_json::json!({"include_usage":true});
            }
            let body_bytes = match serde_json::to_vec(&body) {
                Ok(bytes) => bytes,
                Err(error) => {
                    return Ok(EffectResult::UnsupportedCapability {
                        capability: format!("model payload could not be serialized: {error}"),
                    });
                }
            };
            if body_bytes.len() > 512 * 1024 {
                return Ok(EffectResult::UnsupportedCapability {
                    capability: "request exceeds 512 KiB".into(),
                });
            }
            for attempt in 0..=self.reliability.retry_429 {
                if !self.allowed().await {
                    if attempt == 0 {
                        return Ok(EffectResult::UnsupportedCapability {
                            capability: "HTTP circuit open; no provider request launched".into(),
                        });
                    }
                    // An earlier 429 already reached the provider. Another
                    // concurrent call may open the circuit during backoff.
                    // Never report this as a purely local, refundable denial.
                    return Err(
                        "circuit opened during retry after provider dispatch; charge unknown"
                            .into(),
                    );
                }
                let response = self
                    .client
                    .post(&self.url)
                    .bearer_auth(&self.api_key)
                    .header(reqwest::header::CONTENT_TYPE, "application/json")
                    .body(body_bytes.clone())
                    .send()
                    .await;
                let response = match response {
                    Ok(r) => r,
                    Err(error) => {
                        self.failure().await;
                        // Transport can fail after delivery: never silently retry.
                        return Err(format!("transport_unknown_charge: {error}"));
                    }
                };
                let status = response.status();
                if status.as_u16() == 429 && attempt < self.reliability.retry_429 {
                    // An explicitly retryable response is not a completed
                    // failed attempt for circuit accounting; only exhaustion
                    // or non-retryable failure increments the breaker.
                    // Only HTTP 429 is treated as explicitly non-admitted.
                    // Do not retry timeout, 500, 503, or partial streams.
                    tokio::time::sleep(self.reliability.retry_delay).await;
                    continue;
                }
                if !status.is_success() {
                    self.failure().await;
                    return Err(format!(
                        "http_status_{}: provider may have charged",
                        status.as_u16()
                    ));
                }
                let value: Result<EffectResult, String> = async {
                    if model.streaming {
                        let mut stream = response;
                        let mut parser = SseAccumulator::new(self.reliability.max_response_bytes);
                        while let Some(chunk) = stream.chunk().await.map_err(|e| e.to_string())? {
                            parser.feed(&chunk)?;
                            if parser.done {
                                break;
                            }
                        }
                        parser.finish(&model.model)
                    } else {
                        let body =
                            bounded_bytes(response, self.reliability.max_response_bytes).await?;
                        let payload: serde_json::Value = serde_json::from_slice(&body)
                            .map_err(|e| format!("invalid JSON completion: {e}"))?;
                        if payload.get("error").is_some_and(|v| !v.is_null()) {
                            return Err("provider returned JSON error body".into());
                        }
                        if has_tool_calls(payload.pointer("/choices/0/message/tool_calls")) {
                            return Err(
                                "tool calls unsupported without an explicitly approved executor"
                                    .into(),
                            );
                        }
                        let content = payload
                            .pointer("/choices/0/message/content")
                            .and_then(|v| v.as_str())
                            .ok_or("missing chat completion content")?;
                        Ok(EffectResult::Model {
                            content: content.into(),
                            effective_model: payload
                                .get("model")
                                .and_then(|v| v.as_str())
                                .unwrap_or(&model.model)
                                .into(),
                            usage: parse_usage(&payload),
                        })
                    }
                }
                .await;
                if value.is_ok() {
                    self.successful().await;
                } else {
                    self.failure().await;
                }
                return value;
            }
            Err("unexpected HTTP retry exit".into())
        })
    }
}

/// Explicit scripted example request, used for independent-world tests.
pub fn sample_request(message: &str) -> EffectSpec {
    EffectSpec::Model(ModelRequest {
        provider: "scripted".into(),
        model: "fixed-v1".into(),
        resolved_model_version: Some("fixed-v1".into()),
        messages: vec![crate::effects::Message {
            role: crate::effects::Role::User,
            content: message.into(),
            tool_call_id: None,
        }],
        sampling: Sampling {
            temperature: Some(0.0),
            top_p: None,
            seed: Some(1),
            max_output_tokens: Some(64),
        },
        tool_schemas: vec![],
        streaming: false,
        response_format: None,
    })
}

#[cfg(test)]
mod effect_integrity_tests {
    use super::*;
    use std::sync::Arc;

    #[tokio::test]
    async fn tampered_model_receipt_fails_strict_replay() {
        let path =
            std::env::temp_dir().join(format!("hiyal-effect-integrity-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        let mut journal = EffectJournal::open(&path).unwrap();
        let request = sample_request("hello");
        let key = request.key("trial", 0, AgentId(0), 0).unwrap();
        journal
            .resolve_all(
                vec![(key, request)],
                Arc::new(ScriptedBackend),
                1,
                Duration::from_secs(2),
                false,
            )
            .await
            .unwrap();
        drop(journal);
        let file = path.join("effects.jsonl");
        let original = std::fs::read_to_string(&file).unwrap();
        let mut value: serde_json::Value = serde_json::from_str(original.trim()).unwrap();
        value["result"]["payload"]["content"] = "forged".into();
        std::fs::write(
            &file,
            format!("{}\n", serde_json::to_string(&value).unwrap()),
        )
        .unwrap();
        assert!(EffectJournal::open_existing(&path).is_err());
        std::fs::remove_dir_all(&path).unwrap();
    }
}

#[cfg(test)]
mod batch_regression_tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Tracking {
        launches: Arc<AtomicUsize>,
    }
    impl Backend for Tracking {
        fn resolve<'a>(&'a self, _: &'a EffectSpec) -> BoxResult<'a> {
            self.launches.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Err("provider connection interrupted after sending".into()) })
        }
    }
    #[test]
    fn standalone_writable_journals_have_an_exclusive_writer_lease() {
        let path = std::env::temp_dir().join(format!("hiyal-journal-lock-{}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        let first = EffectJournal::open(&path).unwrap();
        assert!(EffectJournal::open(&path).is_err());
        assert!(path.join(".effects-writer.lock").exists());
        drop(first);
        assert!(!path.join(".effects-writer.lock").exists());
        let second = EffectJournal::open_existing(&path).unwrap();
        drop(second);
        fs::remove_dir_all(&path).unwrap();
    }
    #[tokio::test]
    async fn duplicate_batch_is_rejected_before_any_provider_launch() {
        let path = std::env::temp_dir().join(format!("hiyal-dupes-{}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        let mut journal = EffectJournal::open(&path).unwrap();
        journal.bind_trial("trial").unwrap();
        let spec = sample_request("one");
        let key = spec.key("trial", 0, AgentId(0), 0).unwrap();
        let launches = Arc::new(AtomicUsize::new(0));
        let result = journal
            .resolve_all(
                vec![(key.clone(), spec.clone()), (key, spec)],
                Arc::new(Tracking {
                    launches: launches.clone(),
                }),
                2,
                Duration::from_secs(1),
                false,
            )
            .await;
        assert!(result.is_err());
        assert_eq!(launches.load(Ordering::SeqCst), 0);
        assert_eq!(journal.receipt_count(), 0);
        drop(journal);
        fs::remove_dir_all(path).unwrap();
    }
    #[tokio::test]
    async fn backend_transport_failure_is_logged_with_unknown_billing() {
        let path = std::env::temp_dir().join(format!("hiyal-billing-{}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        let mut journal = EffectJournal::open(&path).unwrap();
        journal.bind_trial("trial").unwrap();
        let spec = sample_request("perhaps billed");
        let key = spec.key("trial", 0, AgentId(0), 0).unwrap();
        let result = journal
            .resolve_all(
                vec![(key, spec)],
                Arc::new(Tracking {
                    launches: Arc::new(AtomicUsize::new(0)),
                }),
                1,
                Duration::from_secs(1),
                false,
            )
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].billing, BillingDisposition::Unknown);
        drop(journal);
        assert_eq!(EffectJournal::inspect_read_only(&path, "trial").unwrap(), 1);
        assert!(EffectJournal::inspect_read_only(&path, "other").is_err());
        fs::remove_dir_all(path).unwrap();
    }
}

#[cfg(test)]
mod r3_replay_billing_regressions {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static TEST_ID: AtomicU64 = AtomicU64::new(0);
    struct Untrusted;
    impl Backend for Untrusted {
        fn resolve<'a>(&'a self, _: &'a EffectSpec) -> BoxResult<'a> {
            Box::pin(async {
                Ok(EffectResult::UnsupportedCapability {
                    capability: "unsupported after uncertain dispatch".into(),
                })
            })
        }
    }
    fn dir() -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "hiyal-r3-ledger-{}-{}",
            std::process::id(),
            TEST_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        path
    }
    #[tokio::test]
    async fn ambiguous_unsupported_results_remain_unknown_on_disk() {
        let path = dir();
        let mut ledger = EffectJournal::open(&path).unwrap();
        ledger.bind_trial("test").unwrap();
        let spec = sample_request("ambiguous");
        let key = spec.key("test", 0, AgentId(0), 0).unwrap();
        let record = ledger
            .resolve_all(
                vec![(key, spec)],
                Arc::new(Untrusted),
                1,
                Duration::from_millis(500),
                false,
            )
            .await
            .unwrap();
        assert_eq!(record[0].billing, BillingDisposition::Unknown);
        drop(ledger);
        assert_eq!(EffectJournal::inspect_read_only(&path, "test").unwrap(), 1);
        std::fs::remove_dir_all(path).unwrap();
    }
    #[tokio::test]
    async fn historical_oversized_batches_replay_without_launch() {
        let path = dir();
        let mut ledger = EffectJournal::open(&path).unwrap();
        ledger.bind_trial("historic").unwrap();
        let mut items = vec![];
        for i in 0..33 {
            let spec = sample_request("historical");
            let key = spec.key("historic", 0, AgentId(0), i).unwrap();
            ledger
                .resolve_all(
                    vec![(key.clone(), spec.clone())],
                    Arc::new(ScriptedBackend),
                    1,
                    Duration::from_secs(1),
                    false,
                )
                .await
                .unwrap();
            items.push((key, spec));
        }
        assert_eq!(
            ledger
                .resolve_all(
                    items.clone(),
                    Arc::new(ScriptedBackend),
                    1,
                    Duration::from_secs(1),
                    true
                )
                .await
                .unwrap()
                .len(),
            33
        );
        let other = items
            .iter()
            .enumerate()
            .map(|(i, (_, s))| {
                (
                    s.key("historic", 1, AgentId(0), i as u32).unwrap(),
                    s.clone(),
                )
            })
            .collect();
        assert!(
            ledger
                .resolve_all(
                    other,
                    Arc::new(ScriptedBackend),
                    1,
                    Duration::from_secs(1),
                    false
                )
                .await
                .is_err()
        );
        drop(ledger);
        std::fs::remove_dir_all(path).unwrap();
    }
}
#[cfg(all(test, feature = "live-http"))]
mod http_tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    #[test]
    fn parse_live_host_safely_before_sending_bearer_token() {
        assert!(
            OpenAiCompatible::new(
                "http://localhost:8080@attacker.invalid/chat".into(),
                "token".into()
            )
            .is_err()
        );
        assert!(
            OpenAiCompatible::new(
                "http://127.0.0.1.attacker.invalid/chat".into(),
                "token".into()
            )
            .is_err()
        );
        assert!(OpenAiCompatible::new("http://192.168.1.10/chat".into(), "token".into()).is_err());
        assert!(OpenAiCompatible::new("http://127.0.0.1:3333/chat".into(), "token".into()).is_ok());
    }
    #[test]
    fn sse_parser_handles_fragmented_frames_and_rejects_truncation() {
        let mut p = SseAccumulator::new(4096);
        p.feed(b"data: {\"model\":\"v2\",\"choices\":[{\"delta\":{\"content\":\"A\"}}]}\n\n")
            .unwrap();
        p.feed(b"data: {\"choices\":[{\"delta\":{\"content\":\"B\"}}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":2}}\n\n").unwrap();
        p.feed(b"data: [DO").unwrap();
        p.feed(b"NE]\n\n").unwrap();
        assert_eq!(
            p.finish("fallback").unwrap(),
            EffectResult::Model {
                content: "AB".into(),
                effective_model: "v2".into(),
                usage: Some(Usage {
                    input_tokens: Some(4),
                    output_tokens: Some(2),
                    billed_cost: None,
                    currency: None
                }),
            }
        );
        let mut truncated = SseAccumulator::new(4096);
        truncated.feed(b"data: {\"choices\":[]}\n\n").unwrap();
        assert!(truncated.finish("fallback").is_err());
        let mut oversized = SseAccumulator::new(3);
        assert!(oversized.feed(b"data: content\\n").is_err());
    }

    #[tokio::test]
    async fn circuit_opened_during_retry_preserves_unknown_billing_and_reservation() {
        use crate::guard::{BudgetConfig, GovernedBackend};

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let (sent, received) = tokio::sync::oneshot::channel();
        let server = thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            let mut buffer = [0u8; 4096];
            let _ = socket.read(&mut buffer).unwrap();
            write!(
                socket,
                "HTTP/1.1 429 Too Many Requests\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            )
            .unwrap();
            let _ = sent.send(());
        });
        let http = Arc::new(
            OpenAiCompatible::new(format!("http://{addr}/chat"), "secret".into())
                .unwrap()
                .with_reliability(HttpReliability {
                    retry_429: 1,
                    retry_delay: Duration::from_millis(500),
                    circuit_threshold: 1,
                    circuit_cooldown: Duration::from_secs(10),
                    ..HttpReliability::default()
                })
                .unwrap(),
        );
        let governed = Arc::new(
            GovernedBackend::new(http.clone(), 1).with_budget(BudgetConfig {
                maximum_micro_usd: 1_000_000,
                prompt_micro_usd_per_token: 1,
                output_micro_usd_per_token: 1,
                max_input_bytes: 4096,
                max_output_tokens: 2048,
            }),
        );
        let request = sample_request("mid-retry circuit");
        let key = request.key("retry-trial", 0, AgentId(0), 0).unwrap();
        let dir = std::env::temp_dir().join(format!("hiyal-r3-mid-retry-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut journal = EffectJournal::open(&dir).unwrap();
        journal.bind_trial("retry-trial").unwrap();
        let worker = {
            let governed = governed.clone();
            let request = request.clone();
            tokio::spawn(async move {
                journal
                    .resolve_all(
                        vec![(key, request)],
                        governed,
                        1,
                        Duration::from_secs(3),
                        false,
                    )
                    .await
            })
        };
        received.await.unwrap();
        // Model a different in-flight request tripping the shared breaker.
        // The first call has received 429 and is still in its retry backoff.
        tokio::time::sleep(Duration::from_millis(50)).await;
        http.failure().await;

        let receipts = worker.await.unwrap().unwrap();
        assert_eq!(receipts.len(), 1);
        assert_eq!(receipts[0].billing, BillingDisposition::Unknown);
        assert!(matches!(
            &receipts[0].result,
            EffectResult::Failed { category, message }
                if category == "backend" && message.contains("after provider dispatch")
        ));
        assert!(governed.remaining_micro_usd().unwrap() < 1_000_000);
        assert!(matches!(
            governed.resolve(&request).await.unwrap(),
            EffectResult::UnsupportedCapability { capability }
                if capability.contains("quota")
        ));
        server.join().unwrap();
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn retries_only_explicit_429_and_never_500() {
        fn server(
            first_status: &'static str,
            requests: usize,
        ) -> (String, std::thread::JoinHandle<usize>) {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let url = format!("http://{}/completions", listener.local_addr().unwrap());
            let thread = thread::spawn(move || {
                for i in 0..requests {
                    let (mut socket, _) = listener.accept().unwrap();
                    socket
                        .set_read_timeout(Some(Duration::from_secs(5)))
                        .unwrap();
                    let mut bytes = [0u8; 4096];
                    let _ = socket.read(&mut bytes).unwrap();
                    let ok = i > 0;
                    let status = if ok { "200 OK" } else { first_status };
                    let body = if ok {
                        r#"{"choices":[{"message":{"content":"fine"}}],"model":"stub"}"#
                    } else {
                        ""
                    };
                    write!(socket,"HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).unwrap();
                }
                requests
            });
            (url, thread)
        }
        let (url, handle) = server("429 Too Many Requests", 2);
        let http = OpenAiCompatible::new(url, "secret".into())
            .unwrap()
            .with_reliability(HttpReliability {
                retry_429: 1,
                retry_delay: Duration::from_millis(1),
                circuit_threshold: 1,
                ..HttpReliability::default()
            })
            .unwrap();
        assert!(matches!(
            http.resolve(&sample_request("ping")).await.unwrap(),
            EffectResult::Model { .. }
        ));
        assert_eq!(handle.join().unwrap(), 2);

        let (url, handle) = server("500 Server Error", 1);
        let no_retry = OpenAiCompatible::new(url, "secret".into())
            .unwrap()
            .with_reliability(HttpReliability {
                retry_429: 2,
                retry_delay: Duration::from_millis(1),
                circuit_threshold: 1,
                ..HttpReliability::default()
            })
            .unwrap();
        assert!(no_retry.resolve(&sample_request("fail")).await.is_err());
        assert!(
            matches!(no_retry.resolve(&sample_request("fail")).await.unwrap(),
            EffectResult::UnsupportedCapability{capability} if capability.contains("circuit"))
        );
        assert_eq!(handle.join().unwrap(), 1);
    }

    #[test]
    fn null_and_empty_tool_calls_do_not_reject_normal_completions() {
        assert!(!has_tool_calls(None));
        assert!(!has_tool_calls(Some(&serde_json::Value::Null)));
        assert!(!has_tool_calls(Some(&serde_json::json!([]))));
        assert!(has_tool_calls(Some(&serde_json::json!([{"id":"call"}]))));
    }

    #[tokio::test]
    async fn journal_timeout_records_unknown_billing_and_opens_live_circuit() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            let mut packet = [0u8; 4096];
            let _ = socket.read(&mut packet).unwrap();
            thread::sleep(Duration::from_millis(250));
            let _ = write!(
                socket,
                "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{{}}"
            );
        });
        let live = Arc::new(
            OpenAiCompatible::new(format!("http://{addr}/chat"), "token".into())
                .unwrap()
                .with_reliability(HttpReliability {
                    circuit_threshold: 1,
                    ..HttpReliability::default()
                })
                .unwrap(),
        );
        let dir = std::env::temp_dir().join(format!("hiyal-r3-timeout-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let request = sample_request("slow");
        let key = request.key("timeout-trial", 0, AgentId(0), 0).unwrap();
        let mut journal = EffectJournal::open(&dir).unwrap();
        journal.bind_trial("timeout-trial").unwrap();
        let receipts = journal
            .resolve_all(
                vec![(key, request.clone())],
                live.clone(),
                1,
                Duration::from_millis(100),
                false,
            )
            .await
            .unwrap();
        assert_eq!(receipts[0].billing, BillingDisposition::Unknown);
        assert!(
            matches!(receipts[0].result,EffectResult::Failed{ref category,..} if category=="timeout")
        );
        assert!(matches!(
            live.resolve(&request).await.unwrap(),
            EffectResult::UnsupportedCapability { .. }
        ));
        drop(journal);
        server.join().unwrap();
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn invalid_200_response_opens_circuit_without_retry() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/completions", listener.local_addr().unwrap());
        let thread = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut ignored = [0u8; 4096];
            let _ = stream.read(&mut ignored).unwrap();
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{{}}"
            )
            .unwrap();
        });
        let http = OpenAiCompatible::new(url, "secret".into())
            .unwrap()
            .with_reliability(HttpReliability {
                circuit_threshold: 1,
                ..HttpReliability::default()
            })
            .unwrap();
        assert!(http.resolve(&sample_request("bad")).await.is_err());
        assert!(matches!(
            http.resolve(&sample_request("bad")).await.unwrap(),
            EffectResult::UnsupportedCapability { .. }
        ));
        thread.join().unwrap();
    }

    #[tokio::test]
    async fn stream_over_loopback_collects_fragmented_sse_and_usage() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut received = vec![];
            let mut chunk = [0u8; 4096];
            loop {
                let n = socket.read(&mut chunk).unwrap();
                if n == 0 {
                    break;
                }
                received.extend_from_slice(&chunk[..n]);
                if let Some(header) = received.windows(4).position(|c| c == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(&received[..header]);
                    let len = headers
                        .lines()
                        .find_map(|line| {
                            line.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .and_then(|c| c.trim().parse::<usize>().ok())
                        })
                        .unwrap_or(0);
                    if received.len() >= header + 4 + len {
                        break;
                    }
                }
            }
            let message = concat!(
                "data: {\"model\":\"v2\",\"choices\":[{\"delta\":{\"content\":\"A\"}}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"content\":\"B\"}}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":2}}\n\n",
                "data: [DONE]\n\n"
            );
            write!(socket,"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",message.len(),message).unwrap();
            String::from_utf8(received).unwrap()
        });
        let http = OpenAiCompatible::new(
            format!("http://127.0.0.1:{port}/v1/chat/completions"),
            "test".into(),
        )
        .unwrap()
        .with_reliability(HttpReliability {
            include_stream_usage: true,
            ..HttpReliability::default()
        })
        .unwrap();
        let mut spec = sample_request("stream");
        if let EffectSpec::Model(ref mut m) = spec {
            m.streaming = true;
        }
        assert_eq!(
            http.resolve(&spec).await.unwrap(),
            EffectResult::Model {
                content: "AB".into(),
                effective_model: "v2".into(),
                usage: Some(Usage {
                    input_tokens: Some(2),
                    output_tokens: Some(2),
                    billed_cost: None,
                    currency: None
                }),
            }
        );
        let wire = server.join().unwrap();
        assert!(wire.contains("\"stream\":true"));
        assert!(wire.contains("stream_options"));
    }

    #[test]
    fn provider_error_frame_invalidates_partial_stream() {
        let mut parser = SseAccumulator::new(4096);
        parser
            .feed(
                br#"data: {"choices":[{"delta":{"content":"partial"}}]}

"#,
            )
            .unwrap();
        assert!(
            parser
                .feed(
                    br#"data: {"error":{"message":"overloaded"}}

"#
                )
                .is_err()
        );
        assert!(serde_json::json!({"error":{"message":"bad"},"choices":[{"message":{"content":"bad"}}]})
            .get("error").is_some());
    }

    #[test]
    fn simulated_timeout_is_scoped_to_model_calls() {
        let http = OpenAiCompatible::new("http://127.0.0.1:1/chat".into(), "test".into())
            .unwrap()
            .with_reliability(HttpReliability {
                circuit_threshold: 1,
                ..HttpReliability::default()
            })
            .unwrap();
        let tool = EffectSpec::Tool {
            name: "slow".into(),
            input: serde_json::json!({}),
        };
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let guarded = crate::guard::GovernedBackend::new(Arc::new(http), 5);
            guarded.on_timeout(&tool).await;
            let model = sample_request("check");
            // The backend was never reached by the tool timeout.
            // A still-healthy model backend tries transport, not circuit rejection.
            let outcome = guarded.resolve(&model).await;
            assert!(
                outcome.is_err()
                    || !matches!(outcome.unwrap(),
                EffectResult::UnsupportedCapability {capability} if capability.contains("circuit"))
            );
        });
    }

    #[tokio::test]
    async fn opt_in_http_chat_adapter_parses_model_and_usage_from_loopback() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut request = Vec::new();
            let mut chunk = [0u8; 4096];
            loop {
                let n = stream.read(&mut chunk).unwrap();
                if n == 0 {
                    break;
                }
                request.extend_from_slice(&chunk[..n]);
                if let Some(end) = request.windows(4).position(|bytes| bytes == b"\r\n\r\n") {
                    let header = String::from_utf8_lossy(&request[..end]);
                    let content_len = header
                        .lines()
                        .find_map(|line| {
                            line.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .and_then(|value| value.trim().parse::<usize>().ok())
                        })
                        .unwrap_or(0);
                    if request.len() >= end + 4 + content_len {
                        break;
                    }
                }
            }
            let body = r#"{"model":"stub-model-v2","choices":[{"message":{"role":"assistant","content":"ready"}}],"usage":{"prompt_tokens":3,"completion_tokens":2}}"#;
            write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body).unwrap();
            String::from_utf8(request).unwrap()
        });
        let client = OpenAiCompatible::new(
            format!("http://127.0.0.1:{port}/v1/chat/completions"),
            "test-token".into(),
        )
        .unwrap();
        let result = client.resolve(&sample_request("hello")).await.unwrap();
        match result {
            EffectResult::Model {
                content,
                effective_model,
                usage,
            } => {
                assert_eq!(content, "ready");
                assert_eq!(effective_model, "stub-model-v2");
                assert_eq!(usage.unwrap().input_tokens, Some(3));
            }
            other => panic!("unexpected result {other:?}"),
        }
        let wire = handle.join().unwrap();
        assert!(wire.contains("POST /v1/chat/completions"));
        assert!(wire.contains("\"content\":\"hello\""));
        assert!(wire.contains("\"model\":\"fixed-v1\""));
    }
}
