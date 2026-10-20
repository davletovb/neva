//! Strict opt-in budget and permission boundary. No implicit tool execution.
//! Limits are process-local reservations; they are not provider billing data.
use crate::effects::{EffectCallKey, EffectResult, ModelRequest};
use crate::runtime::{Backend, BoxResult, EffectSpec};
use crate::storage::{digest, read_metadata_limited};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

/// Monetary units are integer micro-USD; a reservation is NEVER refunded on
/// transport failure, because a provider might already have billed the call.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BudgetConfig {
    pub maximum_micro_usd: u64,
    pub prompt_micro_usd_per_token: u64,
    pub output_micro_usd_per_token: u64,
    /// Conservative byte upper bound for unknown tokenizer behavior.
    pub max_input_bytes: usize,
    pub max_output_tokens: u32,
}

impl BudgetConfig {
    pub fn reserve_amount(&self, request: &ModelRequest) -> Result<u64, String> {
        // Serialize *every* caller controlled field, including role and
        // tool-call identifiers. Reserve additional framing bytes for the
        // wire format (chat message wrappers and model API envelope).
        let serialized = serde_json::to_vec(request)
            .map_err(|e| e.to_string())?
            .len();
        let overhead = request
            .messages
            .len()
            .checked_mul(64)
            .and_then(|n| n.checked_add(256))
            .ok_or("prompt overhead overflow")?;
        let total = serialized
            .checked_add(overhead)
            .ok_or("request size overflow")?;
        if total > self.max_input_bytes {
            return Err("prompt exceeds configured byte limit".into());
        }
        let requested = request
            .sampling
            .max_output_tokens
            .ok_or("live requests require a max_output_tokens bound")?;
        if requested == 0 || requested > self.max_output_tokens {
            return Err("output token limit exceeded".into());
        }
        // Bytes are a conservative upper bound for UTF-8 prompt tokenization.
        // Providers may bill extra wrapper/tool tokens; pricing is a declared
        // budget assumption and is NOT a guaranteed provider-side price cap.
        let input = u64::try_from(total).map_err(|_| "input too large")?;
        input
            .checked_mul(self.prompt_micro_usd_per_token)
            .and_then(|v| {
                v.checked_add(u64::from(requested).checked_mul(self.output_micro_usd_per_token)?)
            })
            .ok_or("budget reservation overflow".into())
    }
}

#[derive(Clone, Debug)]
pub struct ToolRule {
    pub max_input_bytes: usize,
    pub requires_approval: bool,
    pub has_external_side_effects: bool,
}

pub trait ToolExecutor: Send + Sync {
    /// Each executor must validate its own full input schema before launch.
    /// The safe default is to forbid all operations.
    fn validate(&self, _name: &str, _input: &serde_json::Value) -> Result<(), String> {
        Err("tool has no registered schema validator".into())
    }
    fn call<'a>(&'a self, name: &'a str, input: &'a serde_json::Value) -> BoxResult<'a>;
}

/// Wrap a backend with a fail-closed token/spend budget and explicit tool
/// allowlist. Approval tokens are hashes of the *complete* requested effect.
pub struct GovernedBackend {
    backend: Arc<dyn Backend>,
    budget: Option<BudgetConfig>,
    remaining: AtomicU64,
    tool_rules: BTreeMap<String, ToolRule>,
    approvals: BTreeSet<String>,
    consumed: Mutex<BTreeSet<String>>,
    executor: Option<Arc<dyn ToolExecutor>>,
    max_total_effects: u64,
    started: AtomicU64,
}

impl GovernedBackend {
    pub fn new(backend: Arc<dyn Backend>, max_total_effects: u64) -> Self {
        Self {
            backend,
            budget: None,
            remaining: AtomicU64::new(0),
            tool_rules: BTreeMap::new(),
            approvals: BTreeSet::new(),
            consumed: Mutex::new(BTreeSet::new()),
            executor: None,
            max_total_effects,
            started: AtomicU64::new(0),
        }
    }
    pub fn with_budget(mut self, config: BudgetConfig) -> Self {
        self.remaining = AtomicU64::new(config.maximum_micro_usd);
        self.budget = Some(config);
        self
    }
    pub fn allow_tool(mut self, name: &str, rule: ToolRule) -> Result<Self, String> {
        if name.is_empty()
            || name.len() > 100
            || !name
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.'))
        {
            return Err("unsafe tool identifier".into());
        }
        self.tool_rules.insert(name.into(), rule);
        Ok(self)
    }
    /// Grant only this trial+attempt+actor+call+content identity.
    pub fn approve_call(
        mut self,
        key: &EffectCallKey,
        effect: &EffectSpec,
    ) -> Result<Self, String> {
        if effect.key(
            &key.trial_id,
            key.logical_step,
            key.agent_id,
            key.call_index,
        )? != *key
        {
            return Err("approval request key does not match action".into());
        }
        self.approvals.insert(digest(&(key, effect))?);
        Ok(self)
    }
    pub fn with_executor(mut self, executor: Arc<dyn ToolExecutor>) -> Self {
        self.executor = Some(executor);
        self
    }
    pub fn remaining_micro_usd(&self) -> Option<u64> {
        self.budget
            .as_ref()
            .map(|_| self.remaining.load(Ordering::SeqCst))
    }
    #[allow(deprecated)] // MSRV 1.89: AtomicU64::try_update requires 1.95
    fn reserve(counter: &AtomicU64, units: u64) -> bool {
        counter
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |now| {
                now.checked_sub(units)
            })
            .is_ok()
    }
    fn admit(&self, key: Option<&EffectCallKey>, spec: &EffectSpec) -> Result<Option<u64>, String> {
        let mut reserved = None;
        match spec {
            EffectSpec::Model(request) => {
                if let Some(config) = &self.budget {
                    let cost = config.reserve_amount(request)?;
                    if !Self::reserve(&self.remaining, cost) {
                        return Err("spend cap exhausted".into());
                    }
                    reserved = Some(cost);
                }
            }
            EffectSpec::Tool { name, input } => {
                let rule = self.tool_rules.get(name).ok_or("tool permission denied")?;
                let input_bytes = serde_json::to_vec(input).map_err(|e| e.to_string())?;
                if input_bytes.len() > rule.max_input_bytes {
                    return Err("tool input exceeds allowed size".into());
                }
                let executor = self
                    .executor
                    .as_ref()
                    .ok_or("no tool executor registered")?;
                executor.validate(name, input)?;
                if rule.requires_approval || rule.has_external_side_effects {
                    let key = key.ok_or("tool requires keyed invocation")?;
                    let token = digest(&(key, spec))?;
                    if !self.approvals.contains(&token) {
                        return Err("tool requires per-call explicit approval".into());
                    }
                    let mut consumed = self
                        .consumed
                        .lock()
                        .map_err(|_| "approval state poisoned")?;
                    if !consumed.insert(token) {
                        return Err("approved tool call already consumed".into());
                    }
                }
            }
        }
        Ok(reserved)
    }
}
impl GovernedBackend {
    #[allow(deprecated)] // MSRV 1.89: retain fetch_update compatibility
    async fn call_effect(
        &self,
        key: Option<&EffectCallKey>,
        spec: &EffectSpec,
    ) -> Result<EffectResult, String> {
        if self
            .started
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |old| {
                if old < self.max_total_effects {
                    old.checked_add(1)
                } else {
                    None
                }
            })
            .is_err()
        {
            return Ok(EffectResult::UnsupportedCapability {
                capability: "effect quota exceeded".into(),
            });
        }
        let reserved = match self.admit(key, spec) {
            Ok(held) => held,
            Err(reason) => {
                self.started.fetch_sub(1, Ordering::SeqCst);
                return Ok(EffectResult::UnsupportedCapability { capability: reason });
            }
        };
        match spec {
            EffectSpec::Model(_) => {
                let result = self.backend.resolve(spec).await;
                if matches!(&result, Ok(EffectResult::UnsupportedCapability { .. }))
                    && self.backend.unsupported_is_pre_dispatch()
                {
                    // A locally refused call consumes neither effect quota nor
                    // the optional monetary reservation. Do not require a
                    // budget for the slot refund.
                    if let Some(cost) = reserved {
                        self.remaining.fetch_add(cost, Ordering::SeqCst);
                    }
                    self.started.fetch_sub(1, Ordering::SeqCst);
                }
                result
            }
            EffectSpec::Tool { name, input } => {
                self.executor
                    .as_ref()
                    .ok_or("tool executor unavailable")?
                    .call(name, input)
                    .await
            }
        }
    }
}
impl Backend for GovernedBackend {
    /// Bind the declared cap to trial files and reconstruct conservative
    /// previously reserved spend while the Trial holds both writer leases.
    fn recover_admission(
        &self,
        dir: &Path,
        receipts: &BTreeMap<EffectCallKey, crate::runtime::Receipt>,
    ) -> Result<(), String> {
        if let Some(config) = &self.budget {
            let file = dir.join("trial-budget.json");
            if file.exists() {
                let saved: BudgetConfig = serde_json::from_slice(&read_metadata_limited(&file)?)
                    .map_err(|e| e.to_string())?;
                if &saved != config {
                    return Err(
                        "trial budget/pricing changed across resume; refuse live dispatch".into(),
                    );
                }
            } else {
                if !receipts.is_empty() {
                    return Err(
                        "existing trial lacks durable budget config; refuse live dispatch".into(),
                    );
                }
                let serialized = serde_json::to_vec(config).map_err(|e| e.to_string())?;
                let mut out = std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&file)
                    .map_err(|e| e.to_string())?;
                use std::io::Write;
                out.write_all(&serialized)
                    .and_then(|_| out.sync_all())
                    .map_err(|e| e.to_string())?;
                crate::storage::sync_trial_directory(dir)?;
            }
            let mut used = 0u64;
            for receipt in receipts.values() {
                if receipt.billing != crate::runtime::BillingDisposition::NotCharged
                    && let EffectSpec::Model(request) = &receipt.spec
                {
                    used = used
                        .checked_add(config.reserve_amount(request)?)
                        .ok_or("durable budget accounting overflow")?;
                }
            }
            if used > config.maximum_micro_usd {
                return Err("recorded provider reservations exceed trial cap".into());
            }
            self.remaining
                .store(config.maximum_micro_usd - used, Ordering::SeqCst);
        }
        // Unsupported/pre-dispatch denials release their admission slot; a
        // NotCharged receipt is evidence of that decision. Charged/Unknown
        // effects consume one slot and are conservatively restored.
        let admitted = receipts
            .values()
            .filter(|r| r.billing != crate::runtime::BillingDisposition::NotCharged)
            .count() as u64;
        if admitted > self.max_total_effects {
            return Err("past dispatched/uncertain effect count exceeds cap".into());
        }
        self.started.store(admitted, Ordering::SeqCst);
        Ok(())
    }
    fn on_timeout<'a>(
        &'a self,
        spec: &'a EffectSpec,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + 'a>> {
        if matches!(spec, EffectSpec::Model(_)) {
            self.backend.on_timeout(spec)
        } else {
            Box::pin(async {})
        }
    }
    fn unsupported_is_pre_dispatch(&self) -> bool {
        self.backend.unsupported_is_pre_dispatch()
    }
    fn billing_disposition(
        &self,
        spec: &EffectSpec,
        result: &EffectResult,
    ) -> crate::runtime::BillingDisposition {
        if matches!(spec, EffectSpec::Model(_)) {
            self.backend.billing_disposition(spec, result)
        } else if matches!(result, EffectResult::UnsupportedCapability { .. }) {
            // Could be a purely local denial OR untrusted remote tool.
            // Without a persisted per-call admission outcome, stay conservative.
            crate::runtime::BillingDisposition::Unknown
        } else {
            crate::runtime::BillingDisposition::Unknown
        }
    }
    fn permits_tools(&self) -> bool {
        true
    }
    fn backend_id(&self) -> &'static str {
        self.backend.backend_id()
    }
    fn resolve<'a>(&'a self, spec: &'a EffectSpec) -> BoxResult<'a> {
        Box::pin(async move { self.call_effect(None, spec).await })
    }
    fn resolve_keyed<'a>(&'a self, key: &'a EffectCallKey, spec: &'a EffectSpec) -> BoxResult<'a> {
        Box::pin(async move { self.call_effect(Some(key), spec).await })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{ScriptedBackend, sample_request};
    struct Echo;
    impl ToolExecutor for Echo {
        fn validate(&self, name: &str, input: &serde_json::Value) -> Result<(), String> {
            if name != "update.record"
                || input.get("id").and_then(|v| v.as_u64()).is_none()
                || input.as_object().is_none_or(|m| m.len() != 1)
            {
                return Err("invalid tool schema".into());
            }
            Ok(())
        }
        fn call<'a>(&'a self, _: &'a str, input: &'a serde_json::Value) -> BoxResult<'a> {
            Box::pin(async move {
                Ok(EffectResult::Tool {
                    output: input.clone(),
                })
            })
        }
    }
    #[tokio::test]
    async fn tools_are_denied_by_default_and_side_effects_require_exact_approval() {
        let spec = EffectSpec::Tool {
            name: "update.record".into(),
            input: serde_json::json!({"id":1}),
        };
        let backend = GovernedBackend::new(Arc::new(ScriptedBackend), 5)
            .with_executor(Arc::new(Echo))
            .allow_tool(
                "update.record",
                ToolRule {
                    max_input_bytes: 128,
                    requires_approval: true,
                    has_external_side_effects: true,
                },
            )
            .unwrap();
        assert!(matches!(
            backend.resolve(&spec).await.unwrap(),
            EffectResult::UnsupportedCapability { .. }
        ));
        let altered = EffectSpec::Tool {
            name: "update.record".into(),
            input: serde_json::json!({"id":2}),
        };
        let key = spec
            .key("approved-trial", 0, crate::core::AgentId(1), 0)
            .unwrap();
        let wrong = spec
            .key("other-trial", 0, crate::core::AgentId(1), 0)
            .unwrap();
        let allowed = backend.approve_call(&key, &spec).unwrap();
        assert!(matches!(
            allowed.resolve_keyed(&key, &altered).await.unwrap(),
            EffectResult::UnsupportedCapability { .. }
        ));
        assert!(matches!(
            allowed.resolve_keyed(&wrong, &spec).await.unwrap(),
            EffectResult::UnsupportedCapability { .. }
        ));
        assert!(matches!(
            allowed.resolve(&spec).await.unwrap(),
            EffectResult::UnsupportedCapability { .. }
        ));
        assert_eq!(
            allowed.resolve_keyed(&key, &spec).await.unwrap(),
            EffectResult::Tool {
                output: serde_json::json!({"id":1})
            }
        );
        assert!(matches!(
            allowed.resolve_keyed(&key, &spec).await.unwrap(),
            EffectResult::UnsupportedCapability { .. }
        ));
    }
    struct PreflightRefusal;
    impl Backend for PreflightRefusal {
        fn unsupported_is_pre_dispatch(&self) -> bool {
            true
        }
        fn resolve<'a>(&'a self, _: &'a EffectSpec) -> BoxResult<'a> {
            Box::pin(async {
                Ok(EffectResult::UnsupportedCapability {
                    capability: "no provider dispatch".into(),
                })
            })
        }
    }
    #[tokio::test]
    async fn budgetless_pre_dispatch_refusals_do_not_exhaust_quota() {
        let guard = GovernedBackend::new(Arc::new(PreflightRefusal), 1);
        let request = sample_request("no send");
        for _ in 0..3 {
            assert!(matches!(
                guard.resolve(&request).await.unwrap(),
                EffectResult::UnsupportedCapability { .. }
            ));
        }
        assert_eq!(guard.started.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn rejected_pre_dispatch_requests_refund_only_admission_reservations() {
        let cfg = BudgetConfig {
            maximum_micro_usd: 10_000,
            prompt_micro_usd_per_token: 1,
            output_micro_usd_per_token: 1,
            max_input_bytes: 4096,
            max_output_tokens: 80,
        };
        let guard = GovernedBackend::new(Arc::new(PreflightRefusal), 4).with_budget(cfg);
        let spec = sample_request("safe");
        for _ in 0..3 {
            assert!(matches!(
                guard.resolve(&spec).await.unwrap(),
                EffectResult::UnsupportedCapability { .. }
            ));
        }
        assert_eq!(guard.remaining_micro_usd(), Some(10_000));
    }
    #[test]
    fn full_message_framing_and_tool_ids_are_budgeted() {
        let mut spec = sample_request("x");
        if let EffectSpec::Model(ref mut m) = spec {
            m.messages[0].tool_call_id = Some("a".repeat(1000));
            let cfg = BudgetConfig {
                maximum_micro_usd: 1_000_000,
                prompt_micro_usd_per_token: 1,
                output_micro_usd_per_token: 1,
                max_input_bytes: 512,
                max_output_tokens: 100,
            };
            assert!(cfg.reserve_amount(m).is_err());
        }
    }

    #[tokio::test]
    async fn local_denials_do_not_consume_execution_slots() {
        let guarded = GovernedBackend::new(Arc::new(ScriptedBackend), 1);
        let refused = EffectSpec::Tool {
            name: "not-authorized".into(),
            input: serde_json::json!({"a":1}),
        };
        for _ in 0..3 {
            assert!(matches!(
                guarded.resolve(&refused).await.unwrap(),
                EffectResult::UnsupportedCapability { .. }
            ));
        }
        let model = sample_request("valid");
        assert!(matches!(
            guarded.resolve(&model).await.unwrap(),
            EffectResult::Model { .. }
        ));
        assert!(matches!(
            guarded.resolve(&model).await.unwrap(),
            EffectResult::UnsupportedCapability { .. }
        ));
    }

    #[tokio::test]
    async fn recorded_side_effect_replays_without_live_tool_permission() {
        let dir = std::env::temp_dir().join(format!("hiyal-r3-tool-replay-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let spec = EffectSpec::Tool {
            name: "update.record".into(),
            input: serde_json::json!({"id":1}),
        };
        let key = spec
            .key("tool-trial", 0, crate::core::AgentId(0), 0)
            .unwrap();
        {
            let mut journal = crate::runtime::EffectJournal::open(&dir).unwrap();
            journal.bind_trial("tool-trial").unwrap();
            let guard = GovernedBackend::new(Arc::new(ScriptedBackend), 5)
                .with_executor(Arc::new(Echo))
                .allow_tool(
                    "update.record",
                    ToolRule {
                        max_input_bytes: 128,
                        requires_approval: true,
                        has_external_side_effects: true,
                    },
                )
                .unwrap()
                .approve_call(&key, &spec)
                .unwrap();
            let receipts = journal
                .resolve_all(
                    vec![(key.clone(), spec.clone())],
                    Arc::new(guard),
                    1,
                    std::time::Duration::from_secs(2),
                    false,
                )
                .await
                .unwrap();
            assert_eq!(
                receipts[0].billing,
                crate::runtime::BillingDisposition::Unknown
            );
            assert_eq!(receipts[0].effective_provider, "tool:update.record");
        }
        {
            let mut journal = crate::runtime::EffectJournal::open_existing(&dir).unwrap();
            journal.bind_trial("tool-trial").unwrap();
            let receipts = journal
                .resolve_all(
                    vec![(key, spec)],
                    Arc::new(ScriptedBackend),
                    1,
                    std::time::Duration::from_secs(2),
                    true,
                )
                .await
                .unwrap();
            assert!(matches!(receipts[0].result, EffectResult::Tool { .. }));
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn reservations_are_monotonic_even_after_backend_error() {
        let config = BudgetConfig {
            maximum_micro_usd: 2_000,
            prompt_micro_usd_per_token: 2,
            output_micro_usd_per_token: 10,
            max_input_bytes: 4096,
            max_output_tokens: 80,
        };
        let guarded = GovernedBackend::new(Arc::new(ScriptedBackend), 3).with_budget(config);
        let mut spec = sample_request("hello");
        if let EffectSpec::Model(ref mut m) = spec {
            m.sampling.max_output_tokens = Some(50);
        }
        let first = guarded.resolve(&spec).await.unwrap();
        assert!(matches!(first, EffectResult::Model { .. }));
        let remaining = guarded.remaining_micro_usd().unwrap();
        assert!(remaining < 2000);
        assert!(matches!(
            guarded.resolve(&spec).await.unwrap(),
            EffectResult::UnsupportedCapability { .. }
        ));
        assert_eq!(remaining, guarded.remaining_micro_usd().unwrap());
    }
}

#[cfg(test)]
mod r6_resume_quota_tests {
    use super::*;
    use crate::{core::AgentId, runtime::{BillingDisposition, Receipt, ScriptedBackend, sample_request}};
    use std::collections::BTreeMap;

    #[test]
    fn not_charged_rejections_do_not_exhaust_resumed_quota() {
        let dir = std::env::temp_dir().join(format!(
            "hiyal-r6-spend-{}-{}", std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let config = BudgetConfig {
            maximum_micro_usd: 1_000_000, prompt_micro_usd_per_token: 1,
            output_micro_usd_per_token: 1,
            max_input_bytes: 50_000, max_output_tokens: 2048
        };
        let spec = sample_request("admission");
        let rejected = Receipt {
            key: spec.key("quota",0,AgentId(0),0).unwrap(),
            spec: spec.clone(),
            result: EffectResult::UnsupportedCapability{capability:"refused".into()},
            billing: BillingDisposition::NotCharged,
            effective_provider: "offline".into(),
            effective_model: None, content_hash: String::new()
        };
        let charged = Receipt {
            key: spec.key("quota",1,AgentId(0),0).unwrap(),
            billing: BillingDisposition::Unknown, ..rejected.clone()
        };
        let mut receipts = BTreeMap::new();
        receipts.insert(rejected.key.clone(), rejected);
        receipts.insert(charged.key.clone(), charged);
        let guard = GovernedBackend::new(Arc::new(ScriptedBackend), 1).with_budget(config.clone());
        guard.recover_admission(&dir, &BTreeMap::new()).unwrap();
        let resumed = GovernedBackend::new(Arc::new(ScriptedBackend),1).with_budget(config.clone());
        resumed.recover_admission(&dir, &receipts).unwrap();
        assert_eq!(resumed.started.load(Ordering::SeqCst), 1);
        assert_eq!(resumed.remaining_micro_usd(),
                   Some(config.maximum_micro_usd - config.reserve_amount(
                       match &spec { EffectSpec::Model(r) => r, _ => unreachable!() }
                   ).unwrap()));
        assert!(GovernedBackend::new(Arc::new(ScriptedBackend),0)
            .with_budget(config).recover_admission(&dir,&receipts).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
