//! Provenance recorded with every phase report, so measurements stay
//! comparable across revisions, toolchains and machines.
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Serialize)]
pub struct Provenance {
    /// Hiyal's own build record: engine version, rustc, target, source
    /// commit and dirty flag, and the SHA-256 of its `Cargo.lock`.
    pub engine: hiyal::storage::BuildProvenance,
    /// SHA-256 of this benchmark crate's `Cargo.lock`, read at run time from
    /// the source checkout the binary was built from.
    pub bench_lock_sha256: Option<String>,
    pub bench_features: Vec<&'static str>,
    pub cpu: Cpu,
    /// `uname -srm`, when available.
    pub kernel: Option<String>,
    pub unix_time: u64,
}

#[derive(Debug, Serialize)]
pub struct Cpu {
    pub model: Option<String>,
    pub logical_cpus: usize,
    /// Selected `/proc/cpuinfo` flags (Linux). A flag shows what the CPU
    /// offers, not which code path ran; see the README on SHA-NI.
    pub flags: Vec<String>,
    /// Whether `sha2` 0.10 selects its SHA-NI backend on this machine: the
    /// same runtime test it performs (`sha`, `sse2`, `ssse3` and `sse4.1`
    /// all detected; see `sha2/src/sha256/x86.rs`). It shows which code path
    /// is chosen, not an instruction trace. `None` off x86-64.
    pub sha2_hardware_backend: Option<bool>,
}

const FLAGS: [&str; 8] = [
    "sha_ni", "avx2", "avx512f", "aes", "bmi2", "ssse3", "sse4_1", "sse4_2",
];

pub fn collect() -> Provenance {
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let field = |name: &str| {
        cpuinfo
            .lines()
            .find(|line| line.starts_with(name))
            .and_then(|line| line.split_once(':'))
            .map(|(_, value)| value.trim().to_owned())
    };
    let flags = field("flags")
        .map(|all| {
            all.split_whitespace()
                .filter(|f| FLAGS.contains(f))
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    #[cfg(target_arch = "x86_64")]
    let sha2_hardware_backend = Some(
        std::is_x86_feature_detected!("sha")
            && std::is_x86_feature_detected!("sse2")
            && std::is_x86_feature_detected!("ssse3")
            && std::is_x86_feature_detected!("sse4.1"),
    );
    #[cfg(not(target_arch = "x86_64"))]
    let sha2_hardware_backend = None;
    let mut bench_features = Vec::new();
    if cfg!(feature = "replay-cache") {
        bench_features.push("replay-cache");
    }
    Provenance {
        engine: hiyal::storage::BuildProvenance::current(),
        bench_lock_sha256: std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.lock"))
            .ok()
            .map(|bytes| hex(&Sha256::digest(bytes))),
        bench_features,
        cpu: Cpu {
            model: field("model name"),
            logical_cpus: std::thread::available_parallelism().map_or(0, usize::from),
            flags,
            sha2_hardware_backend,
        },
        kernel: std::process::Command::new("uname")
            .arg("-srm")
            .output()
            .ok()
            .filter(|out| out.status.success())
            .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_owned()),
        unix_time: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs()),
    }
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
