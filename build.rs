//! Embed the exact compiler identity into benchmark results at build time.
//! Cargo sets RUSTC to the compiler it actually uses, including rustup
//! toolchain overrides; a package minimum is not a valid replacement.
use std::{env, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RUSTC");
    let rustc = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let output = Command::new(&rustc)
        .arg("--version")
        .output()
        .expect("cannot query the build compiler version");
    assert!(
        output.status.success(),
        "Rust compiler --version failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let version = String::from_utf8(output.stdout).expect("Rust compiler version is not UTF-8");
    let version = version.trim();
    assert!(
        !version.is_empty(),
        "Rust compiler version must not be empty"
    );
    println!("cargo:rustc-env=HIYAL_RUSTC_VERSION={version}");
}
