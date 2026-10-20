//! Native CLI: built-in registered worlds, run/resume/inspect/validate.
//! No dynamic loading of external world traits or embedded Python runtime.
use hiyal::{
    builtins::{Counter, Increment, kind, parse_kind},
    core::AgentId,
    experiment::Trial,
    metrics::CsvMetrics,
    observe::JsonlObserver,
    research::{CounterMetrics, Sweep, ensure_observer_health, paired_forks, verify_fork},
    runtime::{Backend, ScriptedBackend},
    scheduler::Scheduling,
    storage::{Durability, Store},
};
use serde::Deserialize;
use std::{env, fs, path::Path, sync::Arc};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    name: String,
    seed: u64,
    actors: u64,
    attempts: u64,
    amount: u64,
    scheduler: Scheduling,
    model: bool,
    model_id: Option<String>,
    #[serde(default)]
    durability: Durability,
}
fn help() -> &'static str {
    "hiyal run <config.toml> <trial-dir> [--live]\n\
     hiyal resume <trial-dir> <attempts> [--live] [--trusted-snapshot] [--checkpoint-durability]\n\
     hiyal inspect <trial-dir>\n\
     hiyal validate <trial-dir>\n\
     hiyal export <trial-dir> <output.jsonl>\n\
     hiyal sweep <sweep.toml> <output-root>\n\
     hiyal fork <parent-trial> <pair-root> <shock> <attempts>\n\
     Live mode requires --features live-http, OPENAI_COMPAT_URL and OPENAI_COMPAT_API_KEY.\n\
     Set HIYAL_LIVE_MAX_MICRO_USD, HIYAL_LIVE_INPUT_MICRO_USD_PER_TOKEN and\n\
     HIYAL_LIVE_OUTPUT_MICRO_USD_PER_TOKEN for a fail-closed process budget.\n\
     HIYAL_LIVE_MAX_EFFECTS defaults to 100 per process (1..=100000).\n"
}
fn backend(live: bool) -> Result<Arc<dyn Backend>, String> {
    if live {
        #[cfg(feature = "live-http")]
        {
            let url = env::var("OPENAI_COMPAT_URL").map_err(|_| "OPENAI_COMPAT_URL not set")?;
            let api =
                env::var("OPENAI_COMPAT_API_KEY").map_err(|_| "OPENAI_COMPAT_API_KEY not set")?;
            // Pricing must be explicitly configured for every live process.
            // Budgets are reservations, not promises of provider settlement.
            let numeric = |name: &str| -> Result<u64, String> {
                env::var(name)
                    .map_err(|_| format!("{name} required for --live"))?
                    .parse::<u64>()
                    .map_err(|_| format!("invalid {name}"))
            };
            let maximum = numeric("HIYAL_LIVE_MAX_MICRO_USD")?;
            let input_price = numeric("HIYAL_LIVE_INPUT_MICRO_USD_PER_TOKEN")?;
            let output_price = numeric("HIYAL_LIVE_OUTPUT_MICRO_USD_PER_TOKEN")?;
            if maximum == 0 || input_price == 0 || output_price == 0 {
                return Err("live spend cap and declared token rates must be positive".into());
            }
            let http = Arc::new(hiyal::runtime::OpenAiCompatible::new(url, api)?);
            let max_effects = env::var("HIYAL_LIVE_MAX_EFFECTS")
                .unwrap_or_else(|_| "100".into())
                .parse::<u64>()
                .map_err(|_| "invalid HIYAL_LIVE_MAX_EFFECTS")?;
            if !(1..=100_000).contains(&max_effects) {
                return Err("HIYAL_LIVE_MAX_EFFECTS must be 1..=100000".into());
            }
            let guard = hiyal::guard::GovernedBackend::new(http, max_effects).with_budget(
                hiyal::guard::BudgetConfig {
                    maximum_micro_usd: maximum,
                    prompt_micro_usd_per_token: input_price,
                    output_micro_usd_per_token: output_price,
                    max_input_bytes: 64 * 1024,
                    max_output_tokens: 2048,
                },
            );
            return Ok(Arc::new(guard));
        }
        #[cfg(not(feature = "live-http"))]
        {
            return Err("live-http Cargo feature is disabled".into());
        }
    }
    Ok(Arc::new(ScriptedBackend))
}
fn read_kind(path: &Path) -> Result<String, String> {
    let raw = hiyal::storage::read_metadata_limited(&path.join("manifest.json"))?;
    let json: serde_json::Value = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
    json.get("world_kind")
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .ok_or("missing world_kind".into())
}
/// Export exactly one read-only verified event-log prefix. Stage in the same
/// destination directory and publish by hard-linking (create-new semantics),
/// so a failure or concurrent exporter cannot leave a partial public file.
fn marker_actively_locked(path: &Path) -> Result<bool, String> {
    if !path.exists() {
        return Ok(false);
    }
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    // Marker files survive crashes. Only OS lock contention is evidence of
    // an active writer; the OS releases locks on SIGKILL/process exit.
    Ok(file.try_lock().is_err())
}

fn export_trial(dir: &Path, output: &Path) -> Result<(), String> {
    use std::io::Write;

    let verified = Store::<Counter, u64>::inspect_read_only(dir)?;
    hiyal::runtime::EffectJournal::inspect_read_only(dir, &verified.genesis.trial_id)?;
    let trial_root = fs::canonicalize(dir).map_err(|e| e.to_string())?;
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let destination_root = fs::canonicalize(parent).map_err(|e| e.to_string())?;
    if destination_root.starts_with(&trial_root) || output.exists() {
        return Err("export destination exists or is inside the trial root".into());
    }
    // A process-unique staging name prevents two independent exports from
    // interfering; create_new refuses stale files rather than deleting them.
    static EXPORT_NONCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let nonce = EXPORT_NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let temporary = output.with_extension(format!("hiyal-{}-{nonce}.tmp", std::process::id()));
    let mut created = false;
    let result = (|| -> Result<(), String> {
        let mut staging = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|e| e.to_string())?;
        created = true;
        hiyal::observe::write_summaries::<Counter>(dir, &verified, &mut staging)?;
        staging
            .flush()
            .and_then(|_| staging.sync_all())
            .map_err(|e| e.to_string())?;
        drop(staging);
        // Atomic exclusive publication; rename would overwrite a concurrent
        // exporter on Unix. Both files reside on the same filesystem.
        fs::hard_link(&temporary, output).map_err(|e| e.to_string())?;
        fs::remove_file(&temporary).map_err(|e| e.to_string())?;
        #[cfg(unix)]
        fs::File::open(&destination_root)
            .and_then(|f| f.sync_all())
            .map_err(|e| e.to_string())?;
        Ok(())
    })();
    if result.is_err() && created {
        let _ = fs::remove_file(&temporary);
    }
    result
}

async fn launch() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    let cmd = args.first().ok_or(help())?.as_str();
    match cmd {
        "run" => {
            let config_file = args.get(1).ok_or(help())?;
            let dir = args.get(2).ok_or(help())?;
            let live = args.get(3).is_some_and(|s| s == "--live");
            if args.len() > 4 || (args.len() == 4 && !live) {
                return Err(help().into());
            }
            let cfg: Config =
                toml::from_str(&fs::read_to_string(config_file).map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?;
            if cfg.actors == 0 || cfg.actors > 100_000 {
                return Err("actors must be 1..=100000".into());
            }
            let agents = (0..cfg.actors).map(AgentId).collect();
            let model_id = cfg.model_id.as_deref().unwrap_or("fixed-v1");
            let provider = if live {
                "openai-compatible"
            } else {
                "scripted"
            };
            let mut trial = Trial::create(
                dir,
                &cfg.name,
                cfg.seed,
                &kind(cfg.amount, cfg.model, model_id, provider)?,
                Counter { total: 0 },
                Increment {
                    amount: cfg.amount,
                    model: cfg.model,
                    model_id: model_id.to_owned(),
                    provider: provider.to_owned(),
                },
                backend(live)?,
                agents,
                cfg.scheduler,
            )?;
            trial.store.set_durability(cfg.durability)?;
            trial.observe_with(Arc::new(JsonlObserver::recover(dir)?));
            let accepted = trial.run_attempts(cfg.attempts).await?;
            trial.checkpoint()?;
            ensure_observer_health(&trial.diagnostics)?;
            if live && accepted != cfg.attempts {
                return Err(format!(
                    "live run incomplete: accepted {accepted}/{} requested attempts; inspect trial rejections and budget quota",
                    cfg.attempts
                ));
            }
            println!(
                "accepted={accepted} attempts={} total={} hash={}",
                trial.store.sequence(),
                trial.store.world().total,
                trial.store.chain_head()
            );
        }
        "resume" => {
            let dir = Path::new(args.get(1).ok_or(help())?);
            let attempts = args
                .get(2)
                .ok_or(help())?
                .parse::<u64>()
                .map_err(|e| e.to_string())?;
            let opts = &args[3..];
            let live = opts.iter().any(|o| o == "--live");
            let trusted = opts.iter().any(|o| o == "--trusted-snapshot");
            let checkpoint_durability = opts.iter().any(|o| o == "--checkpoint-durability");
            if opts.len() > 3
                || opts.iter().any(|o| {
                    !matches!(
                        o.as_str(),
                        "--live" | "--trusted-snapshot" | "--checkpoint-durability"
                    )
                })
                || opts.iter().collect::<std::collections::BTreeSet<_>>().len() != opts.len()
            {
                return Err(help().into());
            }
            let kind = read_kind(dir)?;
            let policy = parse_kind(&kind)?;
            if policy.provider
                != if live {
                    "openai-compatible"
                } else {
                    "scripted"
                }
            {
                return Err("resume backend differs from saved trial provider".into());
            }
            let backend = backend(live)?;
            let mut trial = if trusted {
                Trial::resume_trusted(dir, &kind, policy, backend, false)?
            } else {
                Trial::resume(dir, &kind, policy, backend, false)?
            };
            if checkpoint_durability {
                trial.store.set_durability(Durability::AtCheckpoint)?;
            }
            // Derived observability remains independently verifiable; it
            // reconstructs projections from all committed records.
            let observer = if trusted {
                JsonlObserver::recover_trusted(&trial.store)?
            } else {
                JsonlObserver::recover(dir)?
            };
            trial.observe_with(Arc::new(observer));
            let accepted = trial.run_attempts(attempts).await?;
            trial.checkpoint()?;
            ensure_observer_health(&trial.diagnostics)?;
            if live && accepted != attempts {
                return Err(format!(
                    "live resume incomplete: accepted {accepted}/{attempts}; inspect trial rejections, spend and quota"
                ));
            }
            println!(
                "accepted={accepted} attempts={} total={} hash={}",
                trial.store.sequence(),
                trial.store.world().total,
                trial.store.chain_head()
            );
        }
        "sweep" => {
            if args.len() != 3 {
                return Err(help().into());
            }
            let cfg: Sweep =
                toml::from_str(&fs::read_to_string(&args[1]).map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?;
            let results = cfg.run_counter(&args[2]).await?;
            println!(
                "completed={} output={}",
                results.len(),
                Path::new(&args[2]).join("results.csv").display()
            );
        }
        "fork" => {
            if args.len() != 5 {
                return Err(help().into());
            }
            let parent = Path::new(&args[1]);
            let root = Path::new(&args[2]);
            let shock = args[3].parse::<u64>().map_err(|e| e.to_string())?;
            let attempts = args[4].parse::<u64>().map_err(|e| e.to_string())?;
            let source = Store::<Counter, u64>::inspect_read_only(parent)?;
            let kind = source.genesis.world_kind.clone();
            let probe = parse_kind(&kind)?;
            if probe.model || probe.provider != "scripted" {
                return Err("fork CLI supports only deterministic scripted counter worlds".into());
            }
            let control_dir = root.join("control");
            let treatment_dir = root.join("treatment");
            // A crash between publication of the two *complete* arms may
            // leave exactly one. Recreate only when it is a verified,
            // unstarted arm belonging to this exact requested parent.
            if control_dir.exists() != treatment_dir.exists() {
                let (existing, expected_arm, expected_parameters) = if control_dir.exists() {
                    (&control_dir, "control", serde_json::json!({}))
                } else {
                    (
                        &treatment_dir,
                        "treatment",
                        serde_json::json!({"added":shock}),
                    )
                };
                let meta = verify_fork::<Counter>(existing)?;
                let prior = Store::<Counter, u64>::inspect_read_only(existing)?;
                if meta.parent_id != source.genesis.trial_id
                    || meta.parent_head != source.chain_head
                    || meta.parent_sequence != source.sequence
                    || meta.arm != expected_arm
                    || meta.intervention != "counter-shock"
                    || meta.parameters != expected_parameters
                    || prior.sequence != 0
                    || marker_actively_locked(&existing.join(".writer.lock"))?
                    || marker_actively_locked(&existing.join(".effects-writer.lock"))?
                {
                    return Err("partial fork cannot be safely recreated".into());
                }
                fs::remove_dir_all(existing).map_err(|e| e.to_string())?;
            }
            let (mut control, mut treatment) = if control_dir.exists() && treatment_dir.exists() {
                let c = verify_fork::<Counter>(&control_dir)?;
                let t = verify_fork::<Counter>(&treatment_dir)?;
                if c.intervention != "counter-shock"
                    || t.intervention != "counter-shock"
                    || c.parent_id != source.genesis.trial_id
                    || t.parent_id != source.genesis.trial_id
                    || c.parent_head != source.chain_head
                    || t.parent_head != source.chain_head
                    || c.parent_sequence != source.sequence
                    || t.parent_sequence != source.sequence
                    || c.parent_head != t.parent_head
                    || c.arm != "control"
                    || t.arm != "treatment"
                    || c.parameters != serde_json::json!({})
                    || t.parameters != serde_json::json!({"added":shock})
                {
                    return Err("paired fork restart attribution mismatch".into());
                }
                (
                    Trial::resume(
                        &control_dir,
                        &kind,
                        parse_kind(&kind)?,
                        Arc::new(ScriptedBackend),
                        false,
                    )?,
                    Trial::resume(
                        &treatment_dir,
                        &kind,
                        parse_kind(&kind)?,
                        Arc::new(ScriptedBackend),
                        false,
                    )?,
                )
            } else {
                paired_forks::<Counter, _, _>(
                    parent,
                    root,
                    "counter-shock",
                    serde_json::json!({"added":shock}),
                    || parse_kind(&kind).expect("validated registered kind"),
                    |world: &mut Counter| {
                        world.total = world.total.checked_add(shock).ok_or("shock overflow")?;
                        Ok(())
                    },
                    Arc::new(ScriptedBackend),
                )?
            };
            if control.store.sequence() > attempts || treatment.store.sequence() > attempts {
                return Err("fork already passed requested attempt count".into());
            }
            control.observe_with(Arc::new(JsonlObserver::recover(&control_dir)?));
            treatment.observe_with(Arc::new(JsonlObserver::recover(&treatment_dir)?));
            control.observe_with(Arc::new(CsvMetrics::recover(&control_dir, CounterMetrics)?));
            treatment.observe_with(Arc::new(CsvMetrics::recover(
                &treatment_dir,
                CounterMetrics,
            )?));
            control
                .run_attempts(attempts - control.store.sequence())
                .await?;
            treatment
                .run_attempts(attempts - treatment.store.sequence())
                .await?;
            ensure_observer_health(&control.diagnostics)?;
            ensure_observer_health(&treatment.diagnostics)?;
            control.checkpoint()?;
            treatment.checkpoint()?;
            println!(
                "control={} treatment={} control_hash={} treatment_hash={}",
                control.store.world().total,
                treatment.store.world().total,
                control.store.chain_head(),
                treatment.store.chain_head()
            );
        }
        "export" => {
            if args.len() != 3 {
                return Err(help().into());
            }
            let dir = Path::new(&args[1]);
            let output = Path::new(&args[2]);
            export_trial(dir, output)?;
            println!("exported redacted event summaries to {}", output.display());
        }
        "inspect" | "validate" => {
            let dir = Path::new(args.get(1).ok_or(help())?);
            if args.len() != 2 {
                return Err(help().into());
            }
            let store = Store::<Counter, u64>::inspect_read_only(dir)?;
            let ledger_count =
                hiyal::runtime::EffectJournal::inspect_read_only(dir, &store.genesis.trial_id)?;
            if cmd == "inspect" {
                println!(
                    "kind={} trial={} events={} committed={} total={} hash={}",
                    store.genesis.world_kind,
                    store.genesis.trial_id,
                    store.sequence,
                    store.logical_step,
                    store.world.total,
                    store.chain_head
                );
            } else {
                // A fork is valid only when its captured, portable source
                // ancestry and intervention attribution also validate.
                let marked_fork = store.genesis.initial_streams.seed_domain.is_some();
                let has_manifest = dir.join("fork.json").exists();
                let has_ancestry = dir.join("ancestry").exists();
                if marked_fork && (!has_manifest || !has_ancestry) {
                    return Err("fork evidence missing from committed branch".into());
                }
                if marked_fork || has_manifest || has_ancestry {
                    verify_fork::<Counter>(dir)?;
                }
                println!(
                    "valid {} events and {} effects; hash={}",
                    store.sequence, ledger_count, store.chain_head
                );
            }
        }
        _ => return Err(help().into()),
    }
    Ok(())
}
#[tokio::main]
async fn main() {
    if let Err(error) = launch().await {
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod export_tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);

    #[tokio::test]
    async fn exported_jsonl_is_valid_verified_and_non_clobbering() {
        let root = env::temp_dir().join(format!(
            "hiyal-r4-export-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let dir = root.join("trial");
        fs::create_dir_all(&root).unwrap();
        {
            let mut trial = Trial::create(
                &dir,
                "export-regression",
                33,
                &kind(1, false, "fixed-v1", "scripted").unwrap(),
                Counter { total: 0 },
                Increment {
                    amount: 1,
                    model: false,
                    model_id: "fixed-v1".into(),
                    provider: "scripted".into(),
                },
                Arc::new(ScriptedBackend),
                vec![AgentId(0)],
                Scheduling::RoundRobin,
            )
            .unwrap();
            trial.run_attempts(3).await.unwrap();
            trial.checkpoint().unwrap();
        }
        let output = root.join("export.jsonl");
        export_trial(&dir, &output).unwrap();
        let bytes = fs::read(&output).unwrap();
        assert_eq!(bytes.iter().filter(|b| **b == b'\n').count(), 3);
        for line in bytes.split(|b| *b == b'\n').filter(|line| !line.is_empty()) {
            let event: hiyal::observe::EventSummary = serde_json::from_slice(line).unwrap();
            assert_eq!(event.kind, "accepted");
        }
        assert!(export_trial(&dir, &output).is_err());
        assert_eq!(fs::read(&output).unwrap(), bytes);
        fs::create_dir_all(dir.join("exports")).unwrap();
        assert!(export_trial(&dir, &dir.join("exports/out.jsonl")).is_err());
        assert!(export_trial(&dir, &dir.join("in-root.jsonl")).is_err());
        assert!(!dir.join("exports/out.jsonl").exists());
        assert!(!dir.join("in-root.jsonl").exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn summary_writer_stops_at_verified_boundary() {
        let root = env::temp_dir().join(format!(
            "hiyal-r4-boundary-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let dir = root.join("trial");
        fs::create_dir_all(&root).unwrap();
        let mut trial = Trial::create(
            &dir,
            "verified-boundary",
            22,
            &kind(1, false, "fixed-v1", "scripted").unwrap(),
            Counter { total: 0 },
            Increment {
                amount: 1,
                model: false,
                model_id: "fixed-v1".into(),
                provider: "scripted".into(),
            },
            Arc::new(ScriptedBackend),
            vec![AgentId(0)],
            Scheduling::RoundRobin,
        )
        .unwrap();
        trial.run_attempts(2).await.unwrap();
        let prior = Store::<Counter, u64>::inspect_read_only(&dir).unwrap();
        trial.run_attempts(2).await.unwrap();
        let mut output = Vec::new();
        hiyal::observe::write_summaries::<Counter>(&dir, &prior, &mut output).unwrap();
        assert_eq!(output.iter().filter(|b| **b == b'\n').count(), 2);
        drop(trial);
        fs::remove_dir_all(root).unwrap();
    }
}
