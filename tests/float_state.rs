//! Worlds and actions holding `f64` must replay exactly. serde_json's default
//! parser can land one ulp away from a shortest-representation float, which
//! breaks the hash chain on resume and inspection, so Hiyal enables its
//! `float_roundtrip` parser itself instead of relying on each consumer.
use hiyal::{
    core::{AgentId, Policy, World},
    experiment::{Rules, Trial},
    runtime::ScriptedBackend,
    scheduler::Scheduling,
    storage::Store,
};
use serde::{Deserialize, Serialize};
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

/// Parses one ulp low without `float_roundtrip`.
const MISPARSED: f64 = 0.40696956400664436;

fn dir(tag: &str) -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    std::env::temp_dir().join(format!(
        "hiyal-float-{}-{}-{}",
        tag,
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Belief {
    value: f64,
}
impl World for Belief {
    type Action = f64;
    type Observation = f64;
    fn observe(&self, _: AgentId) -> Result<f64, String> {
        Ok(self.value)
    }
    fn apply(&mut self, _: AgentId, signal: f64) -> Result<(), String> {
        self.value = 0.9 * self.value + 0.1 * signal;
        Ok(())
    }
}
struct Contrary;
impl Policy<Belief> for Contrary {
    fn propose(&self, _: AgentId, value: &f64) -> Result<f64, String> {
        Ok(1.0 - value)
    }
}

#[test]
fn json_floats_round_trip_exactly() {
    let text = serde_json::to_string(&MISPARSED).unwrap();
    assert_eq!(serde_json::from_str::<f64>(&text).unwrap(), MISPARSED);
}

#[tokio::test]
async fn float_worlds_resume_and_inspect() {
    let p = dir("resume");
    let mut trial = Trial::create(
        &p,
        "float",
        7,
        "float.v1",
        Belief { value: MISPARSED },
        Rules(Contrary),
        Arc::new(ScriptedBackend),
        vec![AgentId(0)],
        Scheduling::RoundRobin,
    )
    .unwrap();
    assert_eq!(trial.run_attempts(25).await.unwrap(), 25);
    trial.checkpoint().unwrap();
    assert_eq!(trial.run_attempts(5).await.unwrap(), 5);
    let live = trial.store.world().clone();
    drop(trial);
    let ro = Store::<Belief, f64>::inspect_read_only(&p).unwrap();
    assert_eq!(ro.world, live);
    let resumed: Trial<Belief, Rules<Contrary>> = Trial::resume(
        &p,
        "float.v1",
        Rules(Contrary),
        Arc::new(ScriptedBackend),
        false,
    )
    .unwrap();
    assert_eq!(resumed.store.world(), &live);
    drop(resumed);
    std::fs::remove_dir_all(p).unwrap();
}
