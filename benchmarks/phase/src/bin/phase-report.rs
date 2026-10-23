//! Fixed phase workloads with provenance, allocation counts and peak heap.
//!
//! phase-report run [--suite baseline|full] [--reps N] [--cold] [--drop-caches]
//!                  [--label TEXT] [--out FILE]
//! phase-report summarize FILE...
//!
//! See benchmarks/phase/README.md for what each phase measures, how cold and
//! warm runs are defined and how to compare two revisions.
use hiyal::storage::{Durability, Store};
use hiyal_phase_bench::{
    AllocStats, ConstantWorld, CountingAlloc, GrowingWorld, KeyedWorld, Scratch, Workload,
    build_journal, dir_bytes, new_trial, per_call_ns, provenance, runtime, stage_event, time_ns,
};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::Path, process::Command};

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc;

/// A fixed workload. The baseline list is versioned: never change an entry
/// in place; add a new suite version instead, so old reports stay comparable.
#[derive(Clone, Copy, Serialize)]
struct Spec {
    shape: &'static str,
    /// Shape-specific size at genesis (slots, history records or tables).
    size: usize,
    /// Committed events in the journal phases.
    events: u64,
}

const BASELINE_V1: &[Spec] = &[
    Spec {
        shape: "constant",
        size: 64,
        events: 200,
    },
    Spec {
        shape: "constant",
        size: 4096,
        events: 200,
    },
    Spec {
        shape: "growing",
        size: 0,
        events: 500,
    },
    Spec {
        shape: "growing",
        size: 2000,
        events: 200,
    },
    Spec {
        shape: "keyed",
        size: 16,
        events: 200,
    },
    Spec {
        shape: "keyed",
        size: 512,
        events: 200,
    },
];

const FULL_EXTRA: &[Spec] = &[
    Spec {
        shape: "constant",
        size: 65536,
        events: 100,
    },
    Spec {
        shape: "growing",
        size: 10000,
        events: 100,
    },
    Spec {
        shape: "growing",
        size: 0,
        events: 2000,
    },
    Spec {
        shape: "keyed",
        size: 4096,
        events: 100,
    },
];

/// Events per sample in both durability phases, so the two run the same
/// history and differ only in durability. Per-event fsync is slow on most
/// disks, so this is short; `commit_buffered_long` covers the workload's full
/// event count, buffered, for scaling with history length.
const COMMIT_EVENTS: u64 = 50;
/// Upper bound on actions per apply sample, which bounds the cloning time
/// spent preparing a sample of a large world.
const APPLY_MAX_ACTIONS: u64 = 20_000;
/// Actions whose allocations are counted for the apply phase.
const APPLY_ALLOC_ACTIONS: u64 = 100;

#[derive(Serialize)]
struct Phase {
    /// What one unit is: a call, an event or an action.
    unit: &'static str,
    /// Nanoseconds per unit, one entry per repetition.
    samples_ns: Vec<f64>,
    median_ns: f64,
    min_ns: f64,
    max_ns: f64,
    /// Allocation counts and peak heap of one further run (per unit).
    /// Absent where the work ran in another process.
    #[serde(skip_serializing_if = "Option::is_none")]
    allocations: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    allocated_bytes: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    peak_bytes: Option<u64>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    notes: BTreeMap<&'static str, Value>,
}

fn phase(unit: &'static str, samples: Vec<f64>, alloc: AllocStats, units: f64) -> Phase {
    let mut p = timing(unit, samples);
    p.allocations = Some(alloc.allocations as f64 / units);
    p.allocated_bytes = Some(alloc.allocated_bytes as f64 / units);
    p.peak_bytes = Some(alloc.peak_bytes);
    p
}

fn timing(unit: &'static str, mut samples: Vec<f64>) -> Phase {
    let mut sorted = samples.clone();
    sorted.sort_by(f64::total_cmp);
    let median = if sorted.is_empty() {
        f64::NAN
    } else if sorted.len() % 2 == 1 {
        sorted[sorted.len() / 2]
    } else {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    };
    samples.shrink_to_fit();
    Phase {
        unit,
        median_ns: median,
        min_ns: sorted.first().copied().unwrap_or(f64::NAN),
        max_ns: sorted.last().copied().unwrap_or(f64::NAN),
        samples_ns: samples,
        allocations: None,
        allocated_bytes: None,
        peak_bytes: None,
        notes: BTreeMap::new(),
    }
}

struct Options {
    reps: usize,
    cold: bool,
    drop_caches: bool,
}

fn measure<W: Workload>(spec: Spec, opts: &Options) -> Result<Value, String>
where
    W::Action: Serialize + DeserializeOwned + Clone + 'static,
{
    let reps = opts.reps.max(1);
    let mut phases = BTreeMap::new();

    // The world as it stands after the workload's events.
    let world = hiyal_phase_bench::apply_actions(&W::build(spec.size), spec.events)?;
    let bytes = hiyal::storage::stable_bytes(&world)?;
    let world_bytes = bytes.len();

    let samples = (0..reps)
        .map(|_| per_call_ns(|| hiyal::storage::stable_bytes(&world)))
        .collect();
    let (alloc, _) = AllocStats::measure(|| hiyal::storage::stable_bytes(&world));
    let mut p = phase("call", samples, alloc, 1.0);
    p.notes.insert("world_bytes", json!(world_bytes));
    phases.insert("serialize", p);

    let samples = (0..reps)
        .map(|_| per_call_ns(|| hiyal::storage::digest(&world)))
        .collect();
    let (alloc, _) = AllocStats::measure(|| hiyal::storage::digest(&world));
    phases.insert("digest", phase("call", samples, alloc, 1.0));

    let samples: Vec<f64> = (0..reps)
        .map(|_| per_call_ns(|| Sha256::digest(&bytes)))
        .collect();
    let (alloc, _) = AllocStats::measure(|| Sha256::digest(&bytes));
    let mut p = phase("call", samples, alloc, 1.0);
    p.notes.insert(
        "mib_per_s",
        json!(world_bytes as f64 / (p.median_ns / 1e9) / (1024.0 * 1024.0)),
    );
    phases.insert("sha256", p);

    // One action on a fresh clone of `world`, applied straight after the
    // clone as Hiyal's staging does; the clone and the drop are untimed, so
    // every timed action acts on exactly the reported world. One clone at a
    // time on purpose: preparing a batch of clones changes the heap layout,
    // which decides whether a growing world's push can extend its buffer in
    // place, and moved the growing-world median by up to 100x.
    let apply_one = |next: &mut W| {
        let step = next.step();
        let actor = step % hiyal_phase_bench::AGENTS;
        next.apply(hiyal::core::AgentId(actor), W::action_for(step, actor))
            .expect("apply");
    };
    let timer_ns = timer_overhead_ns();
    let samples = (0..reps)
        .map(|_| {
            let (mut timed, mut actions) = (0.0, 0u64);
            while timed < hiyal_phase_bench::MIN_SAMPLE.as_nanos() as f64
                && actions < APPLY_MAX_ACTIONS
            {
                let mut next = world.clone();
                timed += time_ns(|| apply_one(&mut next)).0 as f64 - timer_ns;
                actions += 1;
                drop(next);
            }
            timed.max(0.0) / actions as f64
        })
        .collect();
    let mut alloc = AllocStats::default();
    for _ in 0..APPLY_ALLOC_ACTIONS {
        let mut next = world.clone();
        let (one, _) = AllocStats::measure(|| apply_one(&mut next));
        alloc.allocations += one.allocations;
        alloc.allocated_bytes += one.allocated_bytes;
        alloc.peak_bytes = alloc.peak_bytes.max(one.peak_bytes);
    }
    let mut p = phase("action", samples, alloc, APPLY_ALLOC_ACTIONS as f64);
    p.notes.insert(
        "definition",
        json!("one action on a fresh clone of the reported world; clone and drop untimed"),
    );
    p.notes
        .insert("timer_overhead_ns_subtracted", json!(timer_ns));
    phases.insert("apply", p);

    let samples = (0..reps)
        .map(|_| per_call_ns(|| stage_event(&world)))
        .collect();
    let (alloc, _) = AllocStats::measure(|| stage_event(&world));
    phases.insert("stage", phase("event", samples, alloc, 1.0));

    let rt = runtime();
    for (name, durability, events) in [
        ("commit_buffered", Durability::AtCheckpoint, COMMIT_EVENTS),
        ("commit_fsync", Durability::EveryEvent, COMMIT_EVENTS),
        (
            "commit_buffered_long",
            Durability::AtCheckpoint,
            spec.events,
        ),
    ] {
        let mut samples = Vec::new();
        for _ in 0..reps {
            let scratch = Scratch::new(name);
            let mut trial = new_trial::<W>(scratch.path(), spec.size, durability)?;
            let (ns, committed) = time_ns(|| rt.block_on(trial.run_attempts(events)));
            if committed? != events {
                return Err(format!("{name}: not every event committed"));
            }
            samples.push(ns as f64 / events as f64);
        }
        let scratch = Scratch::new(name);
        let mut trial = new_trial::<W>(scratch.path(), spec.size, durability)?;
        let (alloc, committed) = AllocStats::measure(|| rt.block_on(trial.run_attempts(events)));
        committed?;
        let mut p = phase("event", samples, alloc, events as f64);
        p.notes.insert("events", json!(events));
        if name == "commit_buffered_long" {
            p.notes.insert(
                "definition",
                json!("history-length scaling; compare commit_buffered with commit_fsync, not with this"),
            );
        }
        phases.insert(name, p);
    }

    let journal = Scratch::new("journal");
    build_journal::<W>(journal.path(), spec.size, spec.events)?;
    let journal_bytes = std::fs::metadata(journal.path().join("events.jsonl"))
        .map_err(|e| e.to_string())?
        .len();
    let per_event = spec.events as f64;
    let inspect = || Store::<W, W::Action>::inspect_read_only(journal.path());
    let samples = (0..reps)
        .map(|_| time_ns(|| inspect().map(|_| ())).0 as f64 / per_event)
        .collect();
    let (alloc, view) = AllocStats::measure(inspect);
    view?;
    let mut p = phase("event", samples, alloc, per_event);
    p.notes.insert("journal_bytes", json!(journal_bytes));
    p.notes
        .insert("trial_dir_bytes", json!(dir_bytes(journal.path())));
    p.notes.insert(
        "verification",
        json!("full: every event re-applied and every state and chain hash re-checked"),
    );
    phases.insert("replay_full", p);

    #[cfg(feature = "replay-cache")]
    {
        let cached = || Store::<W, W::Action>::inspect_read_only_cached(journal.path());
        let miss: Vec<f64> = (0..reps)
            .map(|_| {
                hiyal::storage::clear_replay_cache();
                time_ns(|| cached().map(|_| ())).0 as f64
            })
            .collect();
        hiyal::storage::clear_replay_cache();
        let (alloc, view) = AllocStats::measure(cached);
        view?;
        phases.insert("inspect_cached_miss", phase("call", miss, alloc, 1.0));
        let hit = (0..reps)
            .map(|_| per_call_ns(|| cached().map(|_| ())))
            .collect();
        let (alloc, view) = AllocStats::measure(cached);
        view?;
        let mut p = phase("call", hit, alloc, 1.0);
        p.notes.insert(
            "verification",
            json!("memo hit: same-bytes evidence already fully replayed in this process"),
        );
        phases.insert("inspect_cached_hit", p);
    }

    let trusted = || Store::<W, W::Action>::open_trusted(journal.path()).map(drop);
    let samples = (0..reps).map(|_| per_call_ns(trusted)).collect();
    let (alloc, opened) = AllocStats::measure(trusted);
    opened?;
    let mut p = phase("call", samples, alloc, 1.0);
    p.notes.insert(
        "verification",
        json!("trusted snapshot: events before the snapshot are NOT re-verified; not comparable to replay_full"),
    );
    phases.insert("open_trusted_snapshot", p);

    if opts.cold {
        let exe = std::env::current_exe().map_err(|e| e.to_string())?;
        let mut samples = Vec::new();
        let mut cache_states = Vec::new();
        for _ in 0..reps {
            cache_states.push(if opts.drop_caches {
                drop_page_cache()
            } else {
                "kept (not requested)".into()
            });
            let out = Command::new(&exe)
                .args(["cold-replay", spec.shape, &journal.path().to_string_lossy()])
                .output()
                .map_err(|e| e.to_string())?;
            if !out.status.success() {
                return Err(String::from_utf8_lossy(&out.stderr).into_owned());
            }
            let ns: f64 = String::from_utf8_lossy(&out.stdout)
                .trim()
                .parse()
                .map_err(|e| format!("cold-replay output: {e}"))?;
            samples.push(ns / per_event);
        }
        let mut p = timing("event", samples);
        p.notes.insert("page_cache", json!(cache_states));
        p.notes.insert(
            "process",
            json!("fresh process per sample: empty replay memo and allocator state"),
        );
        phases.insert("replay_full_cold_process", p);
    }

    Ok(json!({
        "shape": spec.shape,
        "world_kind": W::NAME,
        "size": spec.size,
        "events": spec.events,
        "world_bytes": world_bytes,
        "journal_bytes": journal_bytes,
        "phases": phases,
    }))
}

/// Median cost of an empty `time_ns`, subtracted from each timed action.
fn timer_overhead_ns() -> f64 {
    let mut samples: Vec<u64> = (0..1001).map(|_| time_ns(|| ()).0).collect();
    samples.sort_unstable();
    samples[samples.len() / 2] as f64
}

/// Ask the kernel to drop clean page-cache pages, so the next replay reads
/// from disk. Needs root or passwordless sudo; the outcome is recorded.
fn drop_page_cache() -> String {
    let _ = Command::new("sync").status();
    if std::fs::write("/proc/sys/vm/drop_caches", "3\n").is_ok() {
        return "dropped".into();
    }
    match Command::new("sudo")
        .args(["-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"])
        .output()
    {
        Ok(out) if out.status.success() => "dropped (sudo)".into(),
        Ok(out) => format!(
            "kept: {}",
            String::from_utf8_lossy(&out.stderr)
                .trim()
                .lines()
                .next()
                .unwrap_or("sudo failed")
        ),
        Err(e) => format!("kept: {e}"),
    }
}

fn cold_replay<W: Workload>(dir: &Path) -> Result<u64, String>
where
    W::Action: Serialize + DeserializeOwned + Clone + 'static,
{
    let (ns, view) = time_ns(|| Store::<W, W::Action>::inspect_read_only(dir));
    view?;
    Ok(ns)
}

fn by_shape<T>(
    shape: &str,
    constant: impl FnOnce() -> T,
    growing: impl FnOnce() -> T,
    keyed: impl FnOnce() -> T,
) -> Result<T, String> {
    match shape {
        "constant" => Ok(constant()),
        "growing" => Ok(growing()),
        "keyed" => Ok(keyed()),
        other => Err(format!("unknown shape {other}")),
    }
}

fn run(args: &[String]) -> Result<(), String> {
    let mut suite = "baseline-v1".to_owned();
    let mut opts = Options {
        reps: 5,
        cold: false,
        drop_caches: false,
    };
    let mut label = String::new();
    let mut out = None;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut value = || it.next().cloned().ok_or(format!("{arg} needs a value"));
        match arg.as_str() {
            "--suite" => {
                suite = match value()?.as_str() {
                    "baseline" | "baseline-v1" => "baseline-v1".into(),
                    "full" | "full-v1" => "full-v1".into(),
                    other => return Err(format!("unknown suite {other}")),
                }
            }
            "--reps" => opts.reps = value()?.parse().map_err(|e| format!("--reps: {e}"))?,
            "--cold" => opts.cold = true,
            "--drop-caches" => opts.drop_caches = true,
            "--label" => label = value()?,
            "--out" => out = Some(value()?),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let specs: Vec<Spec> = if suite == "full-v1" {
        BASELINE_V1.iter().chain(FULL_EXTRA).copied().collect()
    } else {
        BASELINE_V1.to_vec()
    };
    let mut workloads = Vec::new();
    for spec in specs {
        eprintln!(
            "phase-report: {} size={} events={}",
            spec.shape, spec.size, spec.events
        );
        workloads.push(by_shape(
            spec.shape,
            || measure::<ConstantWorld>(spec, &opts),
            || measure::<GrowingWorld>(spec, &opts),
            || measure::<KeyedWorld>(spec, &opts),
        )??);
    }
    let report = json!({
        "schema": "hiyal-phase-report/v1",
        "suite": suite,
        "label": label,
        "reps": opts.reps,
        "cold": opts.cold,
        "provenance": provenance::collect(),
        "workloads": workloads,
    });
    let text = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    match out {
        Some(path) => std::fs::write(path, text + "\n").map_err(|e| e.to_string()),
        None => {
            println!("{text}");
            Ok(())
        }
    }
}

/// Median of each phase's medians across reports, per label and workload.
fn summarize(files: &[String]) -> Result<(), String> {
    let mut rows: BTreeMap<(String, String, String), BTreeMap<String, Vec<f64>>> = BTreeMap::new();
    for file in files {
        let report: Value =
            serde_json::from_slice(&std::fs::read(file).map_err(|e| e.to_string())?)
                .map_err(|e| format!("{file}: {e}"))?;
        let label = report["label"].as_str().unwrap_or("").to_owned();
        for w in report["workloads"].as_array().into_iter().flatten() {
            let workload = format!(
                "{} size={} events={}",
                w["shape"].as_str().unwrap_or("?"),
                w["size"],
                w["events"]
            );
            for (phase, p) in w["phases"].as_object().into_iter().flatten() {
                if let Some(m) = p["median_ns"].as_f64() {
                    rows.entry((
                        workload.clone(),
                        phase.clone(),
                        p["unit"].as_str().unwrap_or("").to_owned(),
                    ))
                    .or_default()
                    .entry(label.clone())
                    .or_default()
                    .push(m);
                }
            }
        }
    }
    let labels: Vec<String> = {
        let mut all: Vec<String> = rows.values().flat_map(|m| m.keys().cloned()).collect();
        all.sort();
        all.dedup();
        all
    };
    println!("| workload | phase | unit | {} |", labels.join(" | "));
    println!("|---|---|---|{}", "---|".repeat(labels.len()));
    for ((workload, phase, unit), by_label) in rows {
        let cells: Vec<String> = labels
            .iter()
            .map(|label| {
                by_label.get(label).map_or("–".into(), |values| {
                    let mut v = values.clone();
                    v.sort_by(f64::total_cmp);
                    format!("{:.0} ns (n={})", v[v.len() / 2], v.len())
                })
            })
            .collect();
        println!("| {workload} | {phase} | {unit} | {} |", cells.join(" | "));
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("run") => run(&args[1..]),
        Some("summarize") => summarize(&args[1..]),
        Some("cold-replay") if args.len() == 3 => {
            let dir = Path::new(&args[2]);
            by_shape(
                &args[1],
                || cold_replay::<ConstantWorld>(dir),
                || cold_replay::<GrowingWorld>(dir),
                || cold_replay::<KeyedWorld>(dir),
            )
            .and_then(|r| r)
            .map(|ns| println!("{ns}"))
        }
        _ => Err(
            "usage: phase-report run [--suite baseline|full] [--reps N] [--cold] \
                  [--drop-caches] [--label TEXT] [--out FILE] | summarize FILE..."
                .into(),
        ),
    };
    if let Err(error) = result {
        eprintln!("phase-report: {error}");
        std::process::exit(2);
    }
}
