# Optional R6 fuzz harness

Separate from the workspace to avoid adding libFuzzer to native builds.
Install cargo-fuzz and use a nightly Rust toolchain on an isolated host:

```sh
cd fuzz
cargo +nightly fuzz run memory -- -max_len=1048576
cargo +nightly fuzz run bounded_events -- -max_len=1048576
```

Targets stress validated `MemoryBank` deserialization and bounded newline
reader parsing, including malformed JSON and oversized nested data. Neither
target validates full journal hash-chain or OS-level crash behavior.
A passing compilation or short run is not evidence of comprehensive fuzzing;
record runtime, corpus coverage, crashes and reproducer hashes separately.
No fuzz target is part of ordinary PR CI.
