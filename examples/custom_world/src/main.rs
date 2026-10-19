//! Separately built research world. No Hiyal source changes required.
//! Running without args preserves R0's 6-event standalone smoke test.
use hiyal::{
    core::{AgentId, World},
    experiment::{EffectPolicy, Trial},
    runtime::{Receipt, ScriptedBackend},
    scheduler::Scheduling,
    storage::Store,
};
use serde::{Deserialize, Serialize};
use std::{env, path::Path, sync::Arc};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Society {
    total_actions: u64,
}
impl World for Society {
    type Action = u64;
    type Observation = u64;
    fn observe(&self, _: AgentId) -> Result<u64, String> {
        Ok(self.total_actions)
    }
    fn apply(&mut self, _: AgentId, action: u64) -> Result<(), String> {
        self.total_actions = self.total_actions.checked_add(action).ok_or("overflow")?;
        Ok(())
    }
}
struct Cooperate;
impl EffectPolicy<Society> for Cooperate {
    fn decide(&self, _: AgentId, _: &u64, _: &[Receipt]) -> Result<u64, String> {
        Ok(1)
    }
}
const KIND: &str = "research.cooperative-society.v1";
fn create(dir: &Path) -> Result<Trial<Society, Cooperate>, String> {
    Trial::create(
        dir,
        "cooperation-demo",
        42,
        KIND,
        Society { total_actions: 0 },
        Cooperate,
        Arc::new(ScriptedBackend),
        vec![AgentId(0), AgentId(1), AgentId(2)],
        Scheduling::RoundRobin,
    )
}
async fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        let path = env::temp_dir().join(format!("hiyal-consumer-{}", std::process::id()));
        if path.exists() {
            return Err("temporary trial directory exists".into());
        }
        {
            let mut trial = create(&path)?;
            assert_eq!(trial.run_attempts(6).await?, 6);
            assert_eq!(trial.store.world().total_actions, 6);
            for event in trial.store.read_events()? {
                println!(
                    "{}",
                    serde_json::to_string(&event).map_err(|e| e.to_string())?
                );
            }
        }
        std::fs::remove_dir_all(&path).map_err(|e| e.to_string())?;
    } else {
        let command = args[0].as_str();
        let path = Path::new(args.get(1).ok_or("missing trial directory")?);
        match command {
            "run" => {
                let steps = args
                    .get(2)
                    .ok_or("missing step count")?
                    .parse::<u64>()
                    .map_err(|e| e.to_string())?;
                let mut trial = create(path)?;
                trial.run_attempts(steps).await?;
                trial.checkpoint()?;
                println!(
                    "{} {}",
                    trial.store.world().total_actions,
                    trial.store.chain_head()
                );
            }
            "resume" => {
                let steps = args
                    .get(2)
                    .ok_or("missing step count")?
                    .parse::<u64>()
                    .map_err(|e| e.to_string())?;
                let mut trial =
                    Trial::resume(path, KIND, Cooperate, Arc::new(ScriptedBackend), false)?;
                trial.run_attempts(steps).await?;
                trial.checkpoint()?;
                println!(
                    "{} {}",
                    trial.store.world().total_actions,
                    trial.store.chain_head()
                );
            }
            "inspect" => {
                let store: Store<Society, u64> = Store::open(path)?;
                println!(
                    "actions={} events={} hash={}",
                    store.world().total_actions,
                    store.sequence(),
                    store.chain_head()
                );
            }
            _ => return Err("use run|resume|inspect <dir> [steps]".into()),
        }
    }
    Ok(())
}
#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("{error}");
        std::process::exit(2);
    }
}
