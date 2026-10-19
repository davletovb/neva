//! Native CLI: built-in registered worlds, run/resume/inspect/validate.
//! No dynamic loading of external world traits or embedded Python runtime.
use hiyal::{
    builtins::{Counter, Increment, kind, parse_kind},
    core::AgentId,
    experiment::Trial,
    metrics::CsvMetrics,
    research::{CounterMetrics, Sweep, paired_forks, resume_fork, verify_fork},
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
     hiyal sweep <sweep.toml> <output-root>\n\
     hiyal fork <parent-trial> <pair-root> <shock> <attempts>\n\
     Live mode requires --features live-http, OPENAI_COMPAT_URL and OPENAI_COMPAT_API_KEY.\n"
}
fn backend(live: bool) -> Result<Arc<dyn Backend>, String> {
    if live {
        #[cfg(feature = "live-http")]
        {
            let url = env::var("OPENAI_COMPAT_URL").map_err(|_| "OPENAI_COMPAT_URL not set")?;
            let api =
                env::var("OPENAI_COMPAT_API_KEY").map_err(|_| "OPENAI_COMPAT_API_KEY not set")?;
            return Ok(Arc::new(hiyal::runtime::OpenAiCompatible::new(url, api)?));
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
            let accepted = trial.run_attempts(cfg.attempts).await?;
            trial.checkpoint()?;
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
            let accepted = trial.run_attempts(attempts).await?;
            trial.checkpoint()?;
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
            let (mut control, mut treatment) = if control_dir.exists() || treatment_dir.exists() {
                if !control_dir.exists() || !treatment_dir.exists() {
                    return Err("partial paired fork: both arms must exist".into());
                }
                let c = verify_fork::<Counter>(&control_dir)?;
                let t = verify_fork::<Counter>(&treatment_dir)?;
                if c.intervention != "counter-shock"
                    || t.intervention != "counter-shock"
                    || c.parent_head != t.parent_head
                    || t.parameters != serde_json::json!({"added":shock})
                {
                    return Err("paired fork restart attribution mismatch".into());
                }
                (
                    resume_fork(
                        &control_dir,
                        parse_kind(&kind)?,
                        Arc::new(ScriptedBackend),
                        false,
                    )?,
                    resume_fork(
                        &treatment_dir,
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
                )?
            };
            if control.store.sequence() > attempts || treatment.store.sequence() > attempts {
                return Err("fork already passed requested attempt count".into());
            }
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
