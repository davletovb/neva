#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|bytes: &[u8]| {
    if bytes.len() > 1024 * 1024 { return; }
    if let Ok(memory) = serde_json::from_slice::<hiyal::memory::MemoryBank>(bytes) {
        assert!(memory.validate().is_ok());
    }
});
