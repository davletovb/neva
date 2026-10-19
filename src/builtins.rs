//! Registered built-in deterministic counter society.
use crate::{
    core::{AgentId, World},
    experiment::EffectPolicy,
    runtime::{EffectSpec, Receipt, sample_request},
};
use serde::{Deserialize, Serialize};

pub const KIND: &str = "builtin.counter";
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Counter {
    pub total: u64,
}
impl World for Counter {
    type Action = u64;
    type Observation = u64;
    fn observe(&self, _: AgentId) -> Result<u64, String> {
        Ok(self.total)
    }
    fn apply(&mut self, _: AgentId, add: u64) -> Result<(), String> {
        self.total = self.total.checked_add(add).ok_or("counter overflow")?;
        Ok(())
    }
}
pub struct Increment {
    pub amount: u64,
    pub model: bool,
    pub model_id: String,
    pub provider: String,
}
impl EffectPolicy<Counter> for Increment {
    fn requests(&self, actor: AgentId, obs: &u64) -> Result<Vec<EffectSpec>, String> {
        if self.model {
            let mut request = sample_request(&format!("actor={} observed={obs}", actor.0));
            let EffectSpec::Model(ref mut message) = request else {
                return Err("scripted request must be a model".into());
            };
            message.model = self.model_id.clone();
            message.resolved_model_version = None;
            message.provider = self.provider.clone();
            Ok(vec![request])
        } else {
            Ok(vec![])
        }
    }
    fn decide(&self, _: AgentId, _: &u64, receipts: &[Receipt]) -> Result<u64, String> {
        if self.model && receipts.len() != 1 {
            return Err("exactly one model result required".into());
        }
        Ok(self.amount)
    }
}
pub fn kind(amount: u64, model: bool, model_id: &str, provider: &str) -> Result<String, String> {
    if model_id.is_empty() || model_id.contains(':') || !matches!(provider, "scripted" | "openai-compatible") {
        return Err("invalid model or provider".into());
    }
    Ok(format!("{KIND}:amount={amount}:model={model}:model_id={model_id}:provider={provider}"))
}
pub fn parse_kind(input: &str) -> Result<Increment, String> {
    let tail = input
        .strip_prefix(&format!("{KIND}:"))
        .ok_or("unsupported built-in world")?;
    let mut parts = tail.split(':');
    let amount = parts
        .next()
        .ok_or("missing amount")?
        .strip_prefix("amount=")
        .ok_or("invalid amount")?
        .parse::<u64>()
        .map_err(|e| e.to_string())?;
    let model = parts
        .next()
        .ok_or("missing model")?
        .strip_prefix("model=")
        .ok_or("invalid model")?
        .parse::<bool>()
        .map_err(|e| e.to_string())?;
    let model_id = parts
        .next()
        .ok_or("missing model_id")?
        .strip_prefix("model_id=")
        .ok_or("invalid model_id")?;
    let provider = parts.next().ok_or("missing provider")?
        .strip_prefix("provider=").ok_or("invalid provider")?;
    if parts.next().is_some() || model_id.is_empty() || !matches!(provider, "scripted" | "openai-compatible") {
        return Err("invalid world kind".into());
    }
    Ok(Increment { amount, model, model_id: model_id.into(), provider: provider.into() })
}
