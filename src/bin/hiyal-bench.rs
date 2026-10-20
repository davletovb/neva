//! R5 native benchmark runner: offline, durable-counter workloads.
//! Intentional distinction: the historical Python scheduler step is NOT
//! semantically identical to a fsynced Rust event-sourced Trial.
use hiyal::{
    builtins::{Counter, Increment, kind},
    core::AgentId,
    experiment::{Step, Trial},
    metrics::CsvMetrics,
    observe::JsonlObserver,
    research::{CounterMetrics, ensure_observer_health},
    runtime::ScriptedBackend,
    scheduler::Scheduling,
};
use serde_json::json;
use std::{env, fs, path::Path, sync::Arc, time::{Instant, SystemTime, UNIX_EPOCH}};

fn rss_bytes() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmRSS:"))?;
    line.split_whitespace().nth(1)?.parse::<u64>().ok()?.checked_mul(1024)
}
fn percentile_ns(values: &[u128], numerator: usize, denominator: usize) -> u128 {
    if values.is_empty() { return 0; }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let index = ((sorted.len() * numerator).div_ceil(denominator)).saturating_sub(1);
    sorted[index]
}
async fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: hiyal-bench <actors:1..1000> <steps:1..100000> <on|off>".into());
    }
    let actors = args[0].parse::<u64>().map_err(|e| e.to_string())?;
    let steps = args[1].parse::<u64>().map_err(|e| e.to_string())?;
    let observe = args[2].as_str();
    if !(1..=1000).contains(&actors) || !(1..=100000).contains(&steps)
        || !matches!(observe, "on" | "off") {
        return Err("out-of-range benchmark arguments".into());
    }
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).map_err(|e| e.to_string())?.as_nanos();
    let root = env::temp_dir().join(format!("hiyal-bench-{}-{nonce}", std::process::id()));
    if root.exists() { return Err("benchmark path collision".into()); }
    let kind = kind(1, false, "fixed-v1", "scripted")?;
    let setup = Instant::now();
    let mut trial = Trial::create(
        &root, "benchmark-counter", 1, &kind, Counter { total: 0 },
        Increment {
            amount: 1, model: false, model_id: "fixed-v1".into(), provider: "scripted".into()
        },
        Arc::new(ScriptedBackend), (0..actors).map(AgentId).collect(),
        Scheduling::RoundRobin,
    )?;
    if observe == "on" {
        trial.observe_with(Arc::new(JsonlObserver::recover(&root)?));
        trial.observe_with(Arc::new(CsvMetrics::recover(&root, CounterMetrics)?));
    }
    let setup_ms = setup.elapsed().as_secs_f64() * 1000.0;
    let mut durations = Vec::with_capacity(steps as usize);
    let mut checkpoints = Vec::new();
    let begun = Instant::now();
    for i in 1..=steps {
        let before = Instant::now();
        if !matches!(trial.step().await?, Step::Accepted(_)) {
            return Err("offline benchmark unexpectedly rejected a step".into());
        }
        durations.push(before.elapsed().as_nanos());
        if i == 1 || i == 100 || i == 500 || i == steps {
            checkpoints.push(json!({
                "steps": i, "rss_current_bytes": rss_bytes(),
                "log_bytes": trial.store.log_bytes()
            }));
        }
    }
    let wall_ms = begun.elapsed().as_secs_f64() * 1000.0;
    ensure_observer_health(&trial.diagnostics)?;
    trial.checkpoint()?;
    let result = json!({
        "schema_version": 1,
        "runtime": "rust",
        "profile": "durable_counter_fsync_per_step",
        "observer": observe,
        "actors": actors,
        "steps": steps,
        "setup_ms": setup_ms,
        "wall_ms": wall_ms,
        "throughput_steps_per_second": 1000.0 * steps as f64 / wall_ms,
        "latency_p50_ms": percentile_ns(&durations, 1, 2) as f64 / 1e6,
        "latency_p95_ms": percentile_ns(&durations, 95, 100) as f64 / 1e6,
        "rss_current_bytes": rss_bytes(),
        "checkpoints": checkpoints,
        "rust_version": option_env!("RUSTC_VERSION"),
        "platform": {"os":env::consts::OS,"arch":env::consts::ARCH},
        "semantics": "disk-backed hash-chained world commits, no provider, observers project JSONL and CSV",
        "comparison_warning": "NOT the same unit of work as the historical Python BasicEnvironment.step; no speedup inference"
    });
    println!("{}", serde_json::to_string(&result).map_err(|e| e.to_string())?);
    drop(trial);
    cleanup(&root)?;
    Ok(())
}
fn cleanup(dir: &Path) -> Result<(), String> {
    fs::remove_dir_all(dir).map_err(|e| e.to_string())
}
#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("{error}");
        std::process::exit(2);
    }
}
