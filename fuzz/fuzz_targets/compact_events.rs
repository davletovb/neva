#![no_main]
use libfuzzer_sys::fuzz_target;
use hiyal::storage::Outcome;
use serde_json::Value;

fuzz_target!(|bytes: &[u8]| {
    if bytes.len() > 1024 * 1024 { return; }
    if let Ok(event) = serde_json::from_slice::<Outcome<Value, Value>>(bytes) {
        let encoded = serde_json::to_value(&event).expect("serializable event");
        if let Outcome::Accepted { .. } = event {
            if let Ok(original) = serde_json::from_slice::<Value>(bytes) {
                let was_present = original.as_object().is_some_and(|o|o.contains_key("world"));
                let is_present = encoded.as_object().is_some_and(|o|o.contains_key("world"));
                assert_eq!(was_present, is_present, "lost legacy null-world presence");
                if was_present {
                    assert_eq!(original["world"], encoded["world"]);
                }
            }
        }
    }
});
