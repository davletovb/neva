//! Build-time provenance for the authoritative trial manifest.
use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RUSTC");
    println!("cargo:rerun-if-env-changed=TARGET");
    println!("cargo:rerun-if-env-changed=GITHUB_SHA");
    let rustc = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let output = Command::new(rustc).arg("--version").output()
        .expect("failed to determine build compiler identity");
    assert!(output.status.success(), "build compiler version failed");
    let rustc_version = String::from_utf8(output.stdout).expect("invalid compiler version encoding");
    println!("cargo:rustc-env=HIYAL_RUSTC_VERSION={}", rustc_version.trim());
    let target = env::var("TARGET").expect("Cargo TARGET missing");
    println!("cargo:rustc-env=HIYAL_BUILD_TARGET={target}");

    // A revision is reported only if source identity is verifiable. Building
    // from a vendored crate/archive without a git checkout leaves it unknown.
    let dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("Cargo manifest directory"));
    let source = env::var("GITHUB_SHA").ok().filter(|s| s.len() == 40 && s.bytes().all(|b| b.is_ascii_hexdigit()))
        .or_else(|| {
            let root = Command::new("git").args(["rev-parse","--show-toplevel"])
                .current_dir(&dir).output().ok()?;
            if !root.status.success() { return None; }
            let path = PathBuf::from(String::from_utf8(root.stdout).ok()?.trim());
            if path.canonicalize().ok()? != dir.canonicalize().ok()? { return None; }
            let rev = Command::new("git").args(["rev-parse","HEAD"]).current_dir(&dir).output().ok()?;
            if !rev.status.success() { return None; }
            let revision=String::from_utf8(rev.stdout).ok()?.trim().to_string();
            if revision.len()==40 && revision.bytes().all(|b|b.is_ascii_hexdigit()) {
                Some(revision)
            } else { None }
        });
    if let Some(sha) = source {
        println!("cargo:rustc-env=HIYAL_SOURCE_REVISION={sha}");
    }
}
