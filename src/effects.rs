//! Versioned effect data contracts, not an effect executor.
//! Canonical request hashing, replay and live backends remain R1 work.

use crate::core::{AgentId, decimal_u64};
use serde::{Deserialize, Serialize};

/// Checked optional float: serde_json silently converts NaN/Inf to null by
/// default, which would conflate invalid sampling with provider defaults.
mod finite_option_f64 {
    use serde::{
        Deserialize, Deserializer, Serialize, Serializer, de::Error as _, ser::Error as _,
    };

    pub fn serialize<S: Serializer>(
        value: &Option<f64>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        if value.is_some_and(|number| !number.is_finite()) {
            return Err(S::Error::custom("non-finite model sampling parameter"));
        }
        value.serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<f64>, D::Error> {
        let value = Option::<f64>::deserialize(deserializer)?;
        if value.is_some_and(|number| !number.is_finite()) {
            return Err(D::Error::custom("non-finite model sampling parameter"));
        }
        Ok(value)
    }
}

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
    #[serde(with = "finite_option_f64")]
    pub temperature: Option<f64>,
    #[serde(with = "finite_option_f64")]
    pub top_p: Option<f64>,
    #[serde(with = "decimal_u64::option")]
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
    #[serde(with = "decimal_u64")]
    pub logical_step: u64,
    pub agent_id: AgentId,
    pub call_index: u32,
    /// RFC 8785 request hash to be produced by the R1 effect runner.
    pub canonical_request_hash: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "payload", rename_all = "snake_case")]
pub enum EffectRequest {
    Model {
        key: EffectCallKey,
        request: ModelRequest,
    },
    Tool {
        key: EffectCallKey,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    #[serde(with = "decimal_u64::option")]
    pub input_tokens: Option<u64>,
    #[serde(with = "decimal_u64::option")]
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
    Tool {
        output: serde_json::Value,
    },
    UnsupportedCapability {
        capability: String,
    },
    Failed {
        category: String,
        message: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sampling() -> Sampling {
        Sampling {
            temperature: Some(0.7),
            top_p: None,
            seed: Some(u64::MAX),
            max_output_tokens: Some(128),
        }
    }

    #[test]
    fn invalid_sampling_fails_serialization_instead_of_becoming_null() {
        let mut request = sampling();
        request.temperature = Some(f64::NAN);
        assert!(serde_json::to_string(&request).is_err());
        request.temperature = Some(1.0);
        request.top_p = Some(f64::INFINITY);
        assert!(serde_json::to_string(&request).is_err());
        request.top_p = Some(f64::NEG_INFINITY);
        assert!(serde_json::to_string(&request).is_err());
        request.top_p = None;
        assert!(serde_json::to_string(&request).is_ok());
    }

    #[test]
    fn large_seed_and_usage_are_lossless_decimal_strings() {
        let value = serde_json::to_value(sampling()).unwrap();
        assert_eq!(value["seed"], u64::MAX.to_string());
        let parsed: Sampling = serde_json::from_value(value).unwrap();
        assert_eq!(parsed.seed, Some(u64::MAX));
        let usage = Usage {
            input_tokens: Some(u64::MAX),
            output_tokens: Some(0),
            billed_cost: Some("0.00".to_owned()),
            currency: Some("USD".to_owned()),
        };
        let out = serde_json::to_value(&usage).unwrap();
        assert_eq!(out["input_tokens"], u64::MAX.to_string());
        assert_eq!(out["output_tokens"], "0");
        assert_eq!(serde_json::from_value::<Usage>(out).unwrap(), usage);
    }

    #[test]
    fn effect_key_step_is_a_decimal_string() {
        let key = EffectCallKey {
            trial_id: "trial".into(),
            logical_step: u64::MAX,
            agent_id: AgentId(u64::MAX),
            call_index: 0,
            canonical_request_hash: "hash".into(),
        };
        let json = serde_json::to_value(&key).unwrap();
        assert_eq!(json["logical_step"], u64::MAX.to_string());
        assert_eq!(json["agent_id"], u64::MAX.to_string());
    }
}
