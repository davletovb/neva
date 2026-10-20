//! Build-time provenance. Never attribute a consuming app's CI SHA to Hiyal.
use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8(out.stdout).ok()?.trim().to_owned())
}
fn watch_git_path(dir: &Path, path: &str) {
    if let Some(value) = git(dir, &["rev-parse", "--git-path", path]) {
        println!("cargo:rerun-if-changed={value}");
    }
}
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=Cargo.lock");
    println!("cargo:rerun-if-env-changed=RUSTC");
    println!("cargo:rerun-if-env-changed=TARGET");
    let compiler = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let output = Command::new(compiler)
        .arg("--version")
        .output()
        .expect("failed to query build compiler version");
    assert!(output.status.success(), "rustc --version failed");
    let version = String::from_utf8(output.stdout).expect("invalid rustc version");
    println!("cargo:rustc-env=HIYAL_RUSTC_VERSION={}", version.trim());
    println!(
        "cargo:rustc-env=HIYAL_BUILD_TARGET={}",
        env::var("TARGET").expect("missing TARGET")
    );

    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("missing manifest dir"));
    // A vendored dependency in someone else's checkout has no verifiable
    // Hiyal source identity. The consumer's GITHUB_SHA is never trusted.
    let is_own_checkout = git(&crate_dir, &["rev-parse", "--show-toplevel"])
        .and_then(|root| PathBuf::from(root).canonicalize().ok())
        == crate_dir.canonicalize().ok();
    if !is_own_checkout {
        return;
    }

    watch_git_path(&crate_dir, "HEAD");
    watch_git_path(&crate_dir, "index");
    watch_git_path(&crate_dir, "packed-refs");
    if let Some(reference) = git(&crate_dir, &["symbolic-ref", "HEAD"]) {
        watch_git_path(&crate_dir, &reference);
    }

    let sha = git(&crate_dir, &["rev-parse", "HEAD"]);
    if let Some(sha) = sha.filter(|s| s.len() == 40 && s.bytes().all(|b| b.is_ascii_hexdigit())) {
        println!("cargo:rustc-env=HIYAL_SOURCE_REVISION={sha}");
        // We report source dirtiness separately. A dirty tree has a real
        // HEAD hash, but that hash cannot identify the uncommitted source.
        if let Some(status) = git(
            &crate_dir,
            &["status", "--porcelain", "--untracked-files=no"],
        ) {
            println!("cargo:rustc-env=HIYAL_SOURCE_DIRTY={}", !status.is_empty());
        }
    }
}
