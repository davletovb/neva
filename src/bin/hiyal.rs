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
    storage::Store,
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
}
fn help() -> &'static str {
    "hiyal run <config.toml> <trial-dir> [--live]\n\
     hiyal resume <trial-dir> <attempts> [--live]\n\
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
    let raw = fs::read(path.join("manifest.json")).map_err(|e| e.to_string())?;
    let json: serde_json::Value = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
    json.get("world_kind")
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .ok_or("missing world_kind".into())
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
            trial.observe_with(Arc::new(JsonlObserver::recover(dir)?));
            let accepted = trial.run_attempts(cfg.attempts).await?;
            trial.checkpoint()?;
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
            let live = args.get(3).is_some_and(|s| s == "--live");
            if args.len() > 4 || (args.len() == 4 && !live) {
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
            let mut trial = Trial::resume(dir, &kind, policy, backend(live)?, false)?;
            trial.observe_with(Arc::new(JsonlObserver::recover(dir)?));
            let accepted = trial.run_attempts(attempts).await?;
            trial.checkpoint()?;
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
                    || existing.join(".writer.lock").exists()
                    || existing.join(".effects-writer.lock").exists()
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
            // Validate the source first; never treat a projection as authoritative.
            let source = Store::<Counter, u64>::inspect_read_only(dir)?;
            hiyal::runtime::EffectJournal::inspect_read_only(dir, &source.genesis.trial_id)?;
            // Do not modify an existing output or write inside the trial root.
            if output.exists()
                || output.parent().is_some_and(|p| {
                    std::fs::canonicalize(p).ok() == std::fs::canonicalize(dir).ok()
                })
            {
                return Err("export destination exists or is inside the trial root".into());
            }
            let mut output_file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(output)
                .map_err(|e| e.to_string())?;
            let source_file =
                std::fs::File::open(dir.join("events.jsonl")).map_err(|e| e.to_string())?;
            use std::io::{BufRead, BufReader, Write};
            for line in BufReader::new(source_file).lines() {
                let line = line.map_err(|e| e.to_string())?;
                let event: hiyal::storage::Entry<Counter, u64> =
                    serde_json::from_str(&line).map_err(|e| e.to_string())?;
                let summary = hiyal::observe::EventSummary::from_entry(&event);
                serde_json::to_writer(&mut output_file, &summary).map_err(|e| e.to_string())?;
                output_file.write_all(b"\\n").map_err(|e| e.to_string())?;
            }
            output_file.sync_all().map_err(|e| e.to_string())?;
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
