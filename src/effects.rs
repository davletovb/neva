//! Versioned effect *data contracts*, not an effect executor.
//! Validation, canonical request hashing, replay and live backends are R1 work.

use crate::core::AgentId;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    pub tool_call_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sampling {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub seed: Option<u64>,
    pub max_output_tokens: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelRequest {
    pub provider: String,
    pub model: String,
    pub resolved_model_version: Option<String>,
    pub messages: Vec<Message>,
    pub sampling: Sampling,
    pub tool_schemas: Vec<serde_json::Value>,
    pub streaming: bool,
    pub response_format: Option<serde_json::Value>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EffectCallKey {
    pub trial_id: String,
    pub logical_step: u64,
    pub agent_id: AgentId,
    pub call_index: u32,
    /// RFC 8785 request hash to be produced by the R1 effect runner.
    pub canonical_request_hash: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "payload", rename_all = "snake_case")]
pub enum EffectRequest {
    Model { key: EffectCallKey, request: ModelRequest },
    Tool { key: EffectCallKey, name: String, input: serde_json::Value },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub billed_cost: Option<String>,
    pub currency: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", content = "payload", rename_all = "snake_case")]
pub enum EffectResult {
    Model {
        content: String,
        effective_model: String,
        usage: Option<Usage>,
    },
    Tool { output: serde_json::Value },
    UnsupportedCapability { capability: String },
    Failed { category: String, message: String },
}
