//! Reading internally and adjacently tagged enums without serde's buffering.
//!
//! serde's derived deserializers for these representations buffer fields in
//! its private `Content` type whenever the tag is not read first, and stable
//! JSON's sorted keys often put it later. With serde_json's
//! `arbitrary_precision` feature, which any crate in a build can switch on
//! through Cargo feature unification, numbers reach that buffer as a private
//! map, so reading an `f64` fails and a `serde_json::Value` field silently
//! becomes a map (davletovb/hiyal#120). These helpers read the record's fields
//! as `serde_json::Value`s straight from the input, which serde_json handles
//! in every feature set, then rewrite the tag into the externally tagged form,
//! which deserializes without buffering. The bytes on disk are unchanged.
use serde::de::{self, DeserializeOwned, Deserializer, MapAccess, Visitor};
use serde_json::{Map, Value};
use std::fmt;

/// An object's fields, rejecting a repeated key as a derived struct does.
pub(crate) fn fields<'de, D: Deserializer<'de>>(d: D) -> Result<Map<String, Value>, D::Error> {
    struct Fields;
    impl<'de> Visitor<'de> for Fields {
        type Value = Map<String, Value>;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("a tagged enum object")
        }
        fn visit_map<M: MapAccess<'de>>(self, mut access: M) -> Result<Self::Value, M::Error> {
            let mut fields = Map::new();
            while let Some(key) = access.next_key::<String>()? {
                let value = access.next_value::<Value>()?;
                if fields.contains_key(&key) {
                    return Err(de::Error::custom(format_args!("duplicate field `{key}`")));
                }
                fields.insert(key, value);
            }
            Ok(fields)
        }
    }
    d.deserialize_map(Fields)
}

fn tag_name<E: de::Error>(fields: &mut Map<String, Value>, tag: &'static str) -> Result<String, E> {
    match fields.remove(tag) {
        Some(Value::String(name)) => Ok(name),
        Some(other) => Err(E::invalid_type(unexpected(&other), &"a variant name")),
        None => Err(E::missing_field(tag)),
    }
}

fn unexpected(value: &Value) -> de::Unexpected<'_> {
    match value {
        Value::Null => de::Unexpected::Unit,
        Value::Bool(b) => de::Unexpected::Bool(*b),
        Value::Number(_) => de::Unexpected::Other("number"),
        Value::String(s) => de::Unexpected::Str(s),
        Value::Array(_) => de::Unexpected::Seq,
        Value::Object(_) => de::Unexpected::Map,
    }
}

/// `{tag: "variant", ...fields}` as `T`, an externally tagged mirror enum
/// whose struct variants hold the remaining fields.
pub(crate) fn internally_tagged<'de, D, T>(d: D, tag: &'static str) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: DeserializeOwned,
{
    let mut fields = fields(d)?;
    let variant = tag_name::<D::Error>(&mut fields, tag)?;
    let mut external = Map::new();
    external.insert(variant, Value::Object(fields));
    serde_json::from_value(Value::Object(external)).map_err(de::Error::custom)
}

/// `{tag: "variant", content: value}` as `T`, an externally tagged mirror enum.
/// Like serde's adjacently tagged form, any other key is ignored.
pub(crate) fn adjacently_tagged<'de, D, T>(
    d: D,
    tag: &'static str,
    content: &'static str,
) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: DeserializeOwned,
{
    let mut fields = fields(d)?;
    let variant = tag_name::<D::Error>(&mut fields, tag)?;
    let value = fields
        .remove(content)
        .ok_or_else(|| de::Error::missing_field(content))?;
    let mut external = Map::new();
    external.insert(variant, value);
    serde_json::from_value(Value::Object(external)).map_err(de::Error::custom)
}

#[cfg(test)]
mod tests {
    use crate::effects::{EffectResult, Usage};
    use crate::rng::Streams;
    use crate::storage::{LegacyWorld, Outcome, stable_bytes};
    use serde::{Deserialize, Serialize, de::DeserializeOwned};
    use serde_json::{Value, json};

    /// The derived readers these replace, with their original attributes.
    #[derive(Debug, Serialize, Deserialize)]
    #[serde(bound(
        serialize = "W: Serialize, A: Serialize",
        deserialize = "W: DeserializeOwned, A: DeserializeOwned"
    ))]
    #[serde(tag = "outcome", rename_all = "snake_case")]
    enum DerivedOutcome<W, A> {
        Accepted {
            action: A,
            #[serde(
                default,
                skip_serializing_if = "LegacyWorld::is_missing",
                deserialize_with = "LegacyWorld::<W>::deserialize_present"
            )]
            world: LegacyWorld<W>,
            state_hash: String,
            #[serde(default, skip_serializing_if = "Option::is_none")]
            before_apply_streams: Option<Streams>,
        },
        Rejected {
            reason: String,
        },
        Control {
            paused: bool,
        },
    }

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    #[serde(tag = "status", content = "payload", rename_all = "snake_case")]
    enum DerivedEffectResult {
        Model {
            content: String,
            effective_model: String,
            usage: Option<Usage>,
        },
        Tool {
            output: Value,
        },
        UnsupportedCapability {
            capability: String,
        },
        Failed {
            category: String,
            message: String,
        },
    }

    /// With `arbitrary_precision`, serde_json accepts numbers beyond f64.
    fn arbitrary_precision() -> bool {
        serde_json::from_str::<Value>("1e400").is_ok()
    }

    /// The new reader must accept exactly what the derived one accepted, with
    /// the same re-serialized bytes, and reject what it rejected. Under
    /// `arbitrary_precision` the derived reader can wrongly reject or alter
    /// numbers (#120), so there only its successes on number-free input count.
    fn same<New: Serialize + DeserializeOwned, Old: Serialize + DeserializeOwned>(text: &str) {
        let new = serde_json::from_str::<New>(text).map(|v| stable_bytes(&v).unwrap());
        let old = serde_json::from_str::<Old>(text).map(|v| stable_bytes(&v).unwrap());
        let has_numbers = text.bytes().any(|b| b.is_ascii_digit());
        match (new, old) {
            (Ok(a), Ok(b)) if !arbitrary_precision() || !has_numbers => {
                assert_eq!(String::from_utf8(a), String::from_utf8(b), "{text}")
            }
            (Err(_), Err(_)) => {}
            (_, _) if arbitrary_precision() && has_numbers => {}
            (a, b) => panic!("{text}: new {a:?} vs derived {b:?}"),
        }
    }

    #[test]
    fn outcome_reader_matches_the_derived_reader() {
        let streams = serde_json::to_value(Streams::new("t".into(), 3)).unwrap();
        let mut cases = vec![
            json!({"outcome": "accepted", "action": 0.40696956400664436, "state_hash": "h"}),
            json!({"outcome": "accepted", "action": 7, "state_hash": "h", "world": {"x": 1.5}}),
            json!({"outcome": "accepted", "action": 7, "state_hash": "h", "world": null}),
            json!({"outcome": "accepted", "action": -1e-300, "state_hash": "h", "before_apply_streams": streams}),
            json!({"outcome": "accepted", "action": 1, "state_hash": "h", "before_apply_streams": null}),
            json!({"outcome": "accepted", "action": 1, "state_hash": "h", "unknown": [1, 2]}),
            json!({"outcome": "accepted", "state_hash": "h"}),
            json!({"outcome": "accepted", "action": "text", "state_hash": "h"}),
            json!({"outcome": "rejected", "reason": "no"}),
            json!({"outcome": "rejected", "reason": 5}),
            json!({"outcome": "control", "paused": true}),
            json!({"outcome": "control"}),
            json!({"outcome": "other", "reason": "x"}),
            json!({"outcome": 3, "reason": "x"}),
            json!({"reason": "x"}),
            json!({}),
            json!([]),
            json!("accepted"),
            json!(null),
        ];
        // Tag first as well as last, as stable JSON's sorted keys put it.
        cases.push(
            serde_json::from_str(r#"{"state_hash":"h","action":2.5,"outcome":"accepted"}"#)
                .unwrap(),
        );
        for case in &cases {
            same::<Outcome<Value, f64>, DerivedOutcome<Value, f64>>(&case.to_string());
            same::<Outcome<Value, Value>, DerivedOutcome<Value, Value>>(&case.to_string());
        }
        for text in [
            r#"{"outcome":"accepted","action":1,"action":2,"state_hash":"h"}"#,
            r#"{"outcome":"rejected","outcome":"rejected","reason":"x"}"#,
            r#"{"action":1,"outcome":"accepted","state_hash":"h"}"#,
            r#"{"outcome":"accepted","action":1,"state_hash":"h""#,
        ] {
            same::<Outcome<Value, f64>, DerivedOutcome<Value, f64>>(text);
        }
    }

    #[test]
    fn effect_result_reader_matches_the_derived_reader() {
        let usage = json!({"input_tokens": "3", "output_tokens": null, "billed_cost": "0.1", "currency": "USD"});
        for case in [
            json!({"status": "model", "payload": {"content": "c", "effective_model": "m", "usage": usage}}),
            json!({"payload": {"content": "c", "effective_model": "m", "usage": null}, "status": "model"}),
            json!({"status": "tool", "payload": {"output": {"n": 1.5, "big": 18446744073709551615u64, "s": "x"}}}),
            json!({"status": "tool", "payload": {"output": [1, -2, 0.1]}}),
            json!({"status": "unsupported_capability", "payload": {"capability": "web"}}),
            json!({"status": "failed", "payload": {"category": "c", "message": "m"}}),
            json!({"status": "failed", "payload": {"category": "c"}}),
            json!({"status": "failed"}),
            json!({"status": "nope", "payload": {}}),
            json!({"payload": {"output": 1}}),
            json!({"status": "tool", "payload": {"output": 1}, "extra": 1}),
            json!({"status": 1, "payload": {}}),
        ] {
            same::<EffectResult, DerivedEffectResult>(&case.to_string());
        }
        same::<EffectResult, DerivedEffectResult>(
            r#"{"status":"tool","status":"tool","payload":{"output":1}}"#,
        );
    }

    /// The #120 failure: numbers must survive a write and read in every
    /// feature set, including `arbitrary_precision`.
    #[test]
    fn numeric_actions_and_tool_outputs_round_trip() {
        let outcome: Outcome<Value, f64> = Outcome::Accepted {
            action: 0.40696956400664436,
            world: LegacyWorld::Present(json!({"belief": 1e-300, "count": 3})),
            state_hash: "h".into(),
            before_apply_streams: Some(Streams::new("t".into(), 9)),
        };
        let bytes = stable_bytes(&outcome).unwrap();
        let read: Outcome<Value, f64> = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(stable_bytes(&read).unwrap(), bytes);
        let Outcome::Accepted { action, .. } = read else {
            panic!("accepted outcome expected");
        };
        assert_eq!(action.to_bits(), 0.40696956400664436f64.to_bits());

        let result = EffectResult::Tool {
            output: json!({"n": 1.5, "big": u64::MAX, "list": [0.1, -2]}),
        };
        let bytes = stable_bytes(&result).unwrap();
        let read: EffectResult = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(read, result);
        assert_eq!(stable_bytes(&read).unwrap(), bytes);
    }
}
