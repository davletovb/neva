#![no_main]
use libfuzzer_sys::fuzz_target;
use std::io::{BufReader, Cursor};

fuzz_target!(|bytes: &[u8]| {
    if bytes.len() > 1024 * 1024 { return; }
    let mut reader = BufReader::new(Cursor::new(bytes));
    let mut record = Vec::new();
    let _ = hiyal::storage::read_record_limited(&mut reader, &mut record);
    assert!(record.len() <= hiyal::storage::MAX_EVENT_RECORD_BYTES);
    if let Ok(_) = serde_json::from_slice::<hiyal::observe::EventSummary>(&record) {
        // Valid observation metadata remains non-authoritative by contract.
    }
});
