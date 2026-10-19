//! An early, Rust-only vertical slice of the Hiyal simulation engine.
//!
//! This crate is intentionally unpublished. Storage, effect execution, replay,
//! seeded random scheduling, and canonical event hashing are **not** implemented
//! in this first slice; see issue #103 and docs/r0-decisions.md.
#![forbid(unsafe_code)]

pub mod core;
pub mod effects;
