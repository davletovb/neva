//! Asynchronous, bounded effect boundary with keyed record/replay and
//! fsynced effect receipts. World reducers never perform model/network I/O.
use crate::core::AgentId;
use crate::effects::{EffectCallKey, EffectResult, ModelRequest, Sampling, Usage};
use crate::storage::{digest, stable_bytes};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    future::Future,
    io::Write,
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
    file: std::fs::File,
}
impl Drop for EffectWriterLock {
    fn drop(&mut self) {
        let _ = self.file.sync_all();
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
            file: lock_file,
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
        let raw = fs::read(&path).map_err(|e| e.to_string())?;
        let length = raw
            .iter()
            .rposition(|c| *c == b'\n')
            .map(|i| i + 1)
            .unwrap_or(0);
        let mut receipts = BTreeMap::new();
        for line in raw[..length]
            .split(|c| *c == b'\n')
            .filter(|x| !x.is_empty())
        {
            let receipt: Receipt =
                serde_json::from_slice(line).map_err(|e| format!("invalid effect ledger: {e}"))?;
            if receipt.content_hash != receipt.compute_hash()? {
                return Err("effect ledger receipt hash mismatch".into());
            }
            let key = &receipt.key;
            if receipt.spec.key(
                &key.trial_id,
                key.logical_step,
                key.agent_id,
                key.call_index,
            )? != *key
                || receipts.insert(key.clone(), receipt).is_some()
            {
                return Err("duplicate/mismatched effect ledger key".into());
            }
        }
        if length < raw.len() {
            let f = OpenOptions::new()
                .write(true)
                .open(&path)
                .map_err(|e| e.to_string())?;
            f.set_len(length as u64)
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
        let raw = fs::read(dir.as_ref().join("effects.jsonl")).map_err(|e| e.to_string())?;
        if !raw.is_empty() && !raw.ends_with(b"\n") {
            return Err("effect log has uncommitted/truncated final record".into());
        }
        let mut keys = std::collections::BTreeSet::new();
        for line in raw.split(|b| *b == b'\n').filter(|row| !row.is_empty()) {
            let receipt: Receipt = serde_json::from_slice(line).map_err(|e| e.to_string())?;
            let key = &receipt.key;
            if key.trial_id != trial
                || receipt.content_hash != receipt.compute_hash()?
                || receipt
                    .spec
                    .key(trial, key.logical_step, key.agent_id, key.call_index)?
                    != *key
                || !keys.insert(key.clone())
            {
                return Err("effect ledger failed read-only trial/integrity check".into());
            }
        }
        Ok(keys.len())
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
                let result = timeout(deadline, backend.resolve(&spec)).await;
                let response = match result {
                    Ok(Ok(value)) => value,
                    Ok(Err(error)) => EffectResult::Failed {
                        category: "backend".into(),
                        message: error,
                    },
                    Err(_) => EffectResult::Failed {
                        category: "timeout".into(),
                        message: "effect deadline exceeded".into(),
                    },
                };
                let backend_kind = backend.backend_id();
                let model = match &response {
                    EffectResult::Model {
                        effective_model, ..
                    } => Some(effective_model.clone()),
                    _ => None,
                };
                let billing = if backend_kind == "scripted"
                    || matches!(&response, EffectResult::UnsupportedCapability { .. })
                {
                    BillingDisposition::NotCharged
                } else if matches!(&response, EffectResult::Model { .. }) {
                    BillingDisposition::Charged
                } else {
                    BillingDisposition::Unknown
                };
                Ok::<_, String>(Receipt {
                    key,
                    spec,
                    result: response,
                    billing,
                    effective_provider: backend_kind.into(),
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

/// Minimal OpenAI-compatible, opt-in JSON chat adapter, HTTP only via reqwest.
/// No network requests occur unless the "live-http" feature and backend are
/// selected explicitly. Provider seed remains best-effort, not D1.
#[cfg(feature = "live-http")]
pub struct OpenAiCompatible {
    client: reqwest::Client,
    url: String,
    api_key: String,
}
#[cfg(feature = "live-http")]
impl OpenAiCompatible {
    pub fn new(url: String, api_key: String) -> Result<Self, String> {
        let parsed = reqwest::Url::parse(&url).map_err(|e| e.to_string())?;
        if !parsed.username().is_empty() || parsed.password().is_some() {
            return Err("userinfo in live URL is forbidden".into());
        }
        let allowed = match parsed.scheme() {
            "https" => true,
            "http" => parsed.host_str().is_some_and(|host| {
                host == "localhost"
                    || host
                        .parse::<std::net::IpAddr>()
                        .is_ok_and(|address| address.is_loopback())
            }),
            _ => false,
        };
        if !allowed {
            return Err("live model URL must be HTTPS or parsed loopback HTTP".into());
        }
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| e.to_string())?;
        Ok(Self {
            client,
            url: parsed.to_string(),
            api_key,
        })
    }
}
#[cfg(feature = "live-http")]
impl Backend for OpenAiCompatible {
    fn backend_id(&self) -> &'static str {
        "openai-compatible"
    }
    fn resolve<'a>(&'a self, spec: &'a EffectSpec) -> BoxResult<'a> {
        Box::pin(async move {
            let model = match spec {
                EffectSpec::Model(m) => m,
                _ => {
                    return Ok(EffectResult::UnsupportedCapability {
                        capability: "HTTP tool".into(),
                    });
                }
            };
            if model.streaming || !model.tool_schemas.is_empty() || model.response_format.is_some()
            {
                return Ok(EffectResult::UnsupportedCapability {
                    capability: "streaming/tool schemas/response format (minimal adapter)".into(),
                });
            }
            let roles: Vec<_> = model
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
                        let call_id = m
                            .tool_call_id
                            .as_ref()
                            .ok_or("tool role message requires tool_call_id")?;
                        message["tool_call_id"] = serde_json::json!(call_id);
                    }
                    Ok::<_, String>(message)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut body = serde_json::json!({"model":model.model,"messages":roles,"stream":false});
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
            let response = self
                .client
                .post(&self.url)
                .bearer_auth(&self.api_key)
                .json(&body)
                .send()
                .await
                .map_err(|e| e.to_string())?;
            if !response.status().is_success() {
                return Err(format!("HTTP backend status {}", response.status()));
            }
            let payload: serde_json::Value = response.json().await.map_err(|e| e.to_string())?;
            let text = payload
                .pointer("/choices/0/message/content")
                .and_then(|v| v.as_str())
                .ok_or("missing chat completion content")?;
            let effective_model = payload
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or(model.model.as_str())
                .to_owned();
            let usage = payload.get("usage").map(|u| Usage {
                input_tokens: u.get("prompt_tokens").and_then(|v| v.as_u64()),
                output_tokens: u.get("completion_tokens").and_then(|v| v.as_u64()),
                billed_cost: None,
                currency: None,
            });
            Ok(EffectResult::Model {
                content: text.into(),
                effective_model,
                usage,
            })
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
