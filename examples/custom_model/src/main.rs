//! Independent offline model + explicitly allowlisted tool, compiled as a
//! separate crate. This demonstrates effects without network or Python.
use hiyal::{
    core::{AgentId, World},
    effects::{EffectResult, ModelRequest, Usage},
    experiment::{EffectPolicy, Trial},
    runtime::{Backend, BillingDisposition, BoxResult, EffectSpec, Receipt, sample_request},
    scheduler::Scheduling,
    storage::Store,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{env, path::Path, sync::Arc};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Village {
    cooperation: u64,
}
impl World for Village {
    type Action = u64;
    type Observation = u64;
    fn observe(&self, _: AgentId) -> Result<u64, String> {
        Ok(self.cooperation)
    }
    fn apply(&mut self, _: AgentId, action: u64) -> Result<(), String> {
        self.cooperation = self
            .cooperation
            .checked_add(action)
            .ok_or("cooperation overflow")?;
        Ok(())
    }
}

struct Decide;
impl EffectPolicy<Village> for Decide {
    fn requests(&self, _actor: AgentId, obs: &u64) -> Result<Vec<EffectSpec>, String> {
        let EffectSpec::Model(mut model) = sample_request(&format!("cooperation={obs}")) else {
            return Err("invalid scripted request".into());
        };
        model.provider = "native-offline".into();
        model.model = "local-council-v1".into();
        model.resolved_model_version = Some("local-council-v1".into());
        Ok(vec![
            EffectSpec::Model(model),
            EffectSpec::Tool {
                name: "allow-cooperation".into(),
                input: json!({"allowed":true}),
            },
        ])
    }
    fn decide(&self, _: AgentId, _: &u64, results: &[Receipt]) -> Result<u64, String> {
        if results.len() != 2 {
            return Err("expected model and tool receipts in policy order".into());
        }
        match (&results[0].result, &results[1].result) {
            (EffectResult::Model { content, .. }, EffectResult::Tool { output })
                if content == "cooperate" && output["allowed"] == true =>
            {
                Ok(1)
            }
            _ => Err("model or tool refused cooperation".into()),
        }
    }
}

struct OfflineCouncil;
impl Backend for OfflineCouncil {
    fn backend_id(&self) -> &'static str {
        "native-offline"
    }
    // Explicit opt-in to this one narrowly allowlisted, side-effect-free tool.
    fn permits_tools(&self) -> bool {
        true
    }
    fn billing_disposition(&self, _: &EffectSpec, _: &EffectResult) -> BillingDisposition {
        BillingDisposition::NotCharged
    }
    fn unsupported_is_pre_dispatch(&self) -> bool {
        true
    }
    fn resolve<'a>(&'a self, request: &'a EffectSpec) -> BoxResult<'a> {
        Box::pin(async move {
            match request {
                EffectSpec::Model(ModelRequest {
                    provider,
                    model,
                    streaming: false,
                    ..
                }) if provider == "native-offline" && model == "local-council-v1" => {
                    Ok(EffectResult::Model {
                        content: "cooperate".into(),
                        effective_model: "local-council-v1".into(),
                        usage: Some(Usage {
                            input_tokens: Some(1),
                            output_tokens: Some(1),
                            billed_cost: Some("0".into()),
                            currency: Some("USD".into()),
                        }),
                    })
                }
                EffectSpec::Tool { name, input }
                    if name == "allow-cooperation" && input == &json!({"allowed":true}) =>
                {
                    Ok(EffectResult::Tool {
                        output: json!({"allowed":true}),
                    })
                }
                _ => Ok(EffectResult::UnsupportedCapability {
                    capability: "not allowlisted".into(),
                }),
            }
        })
    }
}
const KIND: &str = "external.native-council.v1";
fn create(dir: &Path) -> Result<Trial<Village, Decide>, String> {
    Trial::create(
        dir,
        "native-council",
        91,
        KIND,
        Village { cooperation: 0 },
        Decide,
        Arc::new(OfflineCouncil),
        vec![AgentId(0), AgentId(1)],
        Scheduling::RoundRobin,
    )
}
async fn execute() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.as_slice() {
        [cmd, directory, steps] if cmd == "run" || cmd == "resume" => {
            let steps: u64 = steps.parse::<u64>().map_err(|e| e.to_string())?;
            if steps > 1000 {
                return Err("example steps must not exceed 1000".into());
            }
            let path = Path::new(directory);
            let mut trial = if cmd == "run" {
                create(path)?
            } else {
                Trial::resume(path, KIND, Decide, Arc::new(OfflineCouncil), false)?
            };
            if trial.run_attempts(steps).await? != steps {
                return Err("offline policy did not accept all steps".into());
            }
            trial.checkpoint()?;
            println!(
                "cooperation={} head={}",
                trial.store.world().cooperation,
                trial.store.chain_head()
            );
        }
        [cmd, directory] if cmd == "inspect" => {
            let state = Store::<Village, u64>::inspect_read_only(directory)?;
            println!(
                "cooperation={} seq={} hash={}",
                state.world.cooperation, state.sequence, state.chain_head
            );
        }
        [] => {
            let path = env::temp_dir().join(format!("hiyal-native-model-{}", std::process::id()));
            if path.exists() {
                return Err("example temporary directory exists".into());
            }
            {
                let mut trial = create(&path)?;
                trial.run_attempts(4).await?;
                trial.checkpoint()?;
                assert_eq!(trial.store.world().cooperation, 4);
            }
            let state = Store::<Village, u64>::inspect_read_only(&path)?;
            assert_eq!(state.sequence, 4);
            std::fs::remove_dir_all(path).map_err(|e| e.to_string())?;
            println!("native model + allowlisted tool: 4 committed actions");
        }
        _ => return Err("use [run|resume <dir> <steps>] or inspect <dir>".into()),
    }
    Ok(())
}
#[tokio::main]
async fn main() {
    if let Err(error) = execute().await {
        eprintln!("{error}");
        std::process::exit(2);
    }
}
