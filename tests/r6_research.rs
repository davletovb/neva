//! R6 regression tests for large worlds, OS leases, trusted snapshots,
//! stochastic policy/reducer replay, and overlapping two-phase rounds.
use hiyal::{
    core::{AgentId, Policy, World},
    effects::EffectResult,
    experiment::{EffectPolicy, Rules, Step, Trial},
    rng::Streams,
    runtime::{Backend, BoxResult, ScriptedBackend, sample_request},
    scheduler::Scheduling,
    storage::{Durability, Store},
};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

fn dir(tag: &str) -> std::path::PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    std::env::temp_dir().join(format!(
        "hiyal-r6-{}-{}-{}",
        tag,
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}
#[derive(Clone, Serialize, Deserialize)]
struct Society {
    balances: Vec<u64>,
    transcript: Vec<String>,
}
impl World for Society {
    type Action = u64;
    type Observation = u64;
    fn observe(&self, a: AgentId) -> Result<u64, String> {
        self.balances
            .get(a.0 as usize)
            .copied()
            .ok_or("bad agent".into())
    }
    fn apply(&mut self, a: AgentId, amount: u64) -> Result<(), String> {
        let b = self.balances.get_mut(a.0 as usize).ok_or("bad agent")?;
        *b = b.checked_add(amount).ok_or("overflow")?;
        Ok(())
    }
}
struct Gain;
impl Policy<Society> for Gain {
    fn propose(&self, _: AgentId, _: &u64) -> Result<u64, String> {
        Ok(1)
    }
}
fn create_society(p: &Path) -> Trial<Society, Rules<Gain>> {
    Trial::create(
        p,
        "society",
        5,
        "society.v1",
        Society {
            balances: vec![1; 500],
            transcript: vec!["x".repeat(6000); 20],
        },
        Rules(Gain),
        Arc::new(ScriptedBackend),
        (0..500).map(AgentId).collect(),
        Scheduling::RoundRobin,
    )
    .unwrap()
}

#[tokio::test]
async fn compact_events_no_longer_copy_big_world_and_legacy_snapshot_restores() {
    let p = dir("compact");
    let mut trial = create_society(&p);
    assert_eq!(trial.run_attempts(80).await.unwrap(), 80);
    trial.checkpoint().unwrap();
    let bytes = fs::read(p.join("events.jsonl")).unwrap();
    assert!(
        bytes.len() < 1_000_000,
        "world leaked into event payload: {}",
        bytes.len()
    );
    assert!(!std::str::from_utf8(&bytes).unwrap().contains("\"world\":"));
    drop(trial);
    let state = Store::<Society, u64>::inspect_read_only(&p).unwrap();
    assert_eq!(state.sequence, 80);
    let resumed = Trial::resume_trusted(
        &p,
        "society.v1",
        Rules(Gain),
        Arc::new(ScriptedBackend),
        false,
    )
    .unwrap();
    assert_eq!(resumed.store.sequence(), 80);
    drop(resumed);
    fs::remove_dir_all(p).unwrap();
}

#[tokio::test]
async fn snapshot_fast_mode_has_explicit_trust_boundary() {
    let p = dir("trust");
    {
        let mut t = create_society(&p);
        t.run_attempts(8).await.unwrap();
        t.checkpoint().unwrap();
    }
    let original = fs::read(p.join("events.jsonl")).unwrap();
    let mut events: Vec<serde_json::Value> = String::from_utf8(original)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    events[0]["hash"] = serde_json::json!("0".repeat(64));
    let forged = events
        .iter()
        .map(|e| format!("{}\n", serde_json::to_string(e).unwrap()))
        .collect::<String>();
    fs::write(p.join("events.jsonl"), forged).unwrap();
    assert!(Store::<Society, u64>::inspect_read_only(&p).is_err());
    assert!(
        Trial::resume(
            &p,
            "society.v1",
            Rules(Gain),
            Arc::new(ScriptedBackend),
            false
        )
        .is_err()
    );
    let fast = Trial::resume_trusted(
        &p,
        "society.v1",
        Rules(Gain),
        Arc::new(ScriptedBackend),
        false,
    )
    .unwrap();
    assert_eq!(fast.store.sequence(), 8);
    drop(fast);
    fs::remove_dir_all(p).unwrap();
}

#[tokio::test]
async fn durability_checkpoint_and_stale_os_lock_file_reopen() {
    let p = dir("durable");
    {
        let mut t = create_society(&p);
        t.store.set_durability(Durability::AtCheckpoint).unwrap();
        t.run_attempts(9).await.unwrap();
        t.checkpoint().unwrap();
    }
    // A persistent lock marker is NOT a stale lease; the kernel released it.
    assert!(p.join(".writer.lock").exists());
    assert!(p.join(".effects-writer.lock").exists());
    let t = Trial::resume(
        &p,
        "society.v1",
        Rules(Gain),
        Arc::new(ScriptedBackend),
        false,
    )
    .unwrap();
    assert_eq!(t.store.sequence(), 9);
    drop(t);
    fs::remove_dir_all(p).unwrap();
}

#[derive(Clone, Serialize, Deserialize)]
struct RandomWorld {
    total: u64,
}
impl World for RandomWorld {
    type Action = u64;
    type Observation = ();
    fn observe(&self, _: AgentId) -> Result<(), String> {
        Ok(())
    }
    fn apply(&mut self, _: AgentId, action: u64) -> Result<(), String> {
        self.total = self.total.wrapping_add(action);
        Ok(())
    }
    fn apply_with_streams(
        &mut self,
        a: AgentId,
        action: u64,
        rng: &mut Streams,
    ) -> Result<(), String> {
        let draw = rng.draw_u64("world")?;
        self.apply(a, action ^ draw)
    }
}
struct RandomPolicy;
impl EffectPolicy<RandomWorld> for RandomPolicy {
    fn decide(&self, _: AgentId, _: &(), _: &[hiyal::runtime::Receipt]) -> Result<u64, String> {
        Ok(0)
    }
    fn decide_with_streams(
        &self,
        a: AgentId,
        _: &(),
        _: &[hiyal::runtime::Receipt],
        rng: &mut Streams,
    ) -> Result<u64, String> {
        rng.draw_u64(&format!("agent/{}", a.0))
    }
}
#[tokio::test]
async fn named_streams_for_policy_and_world_replay_bitwise() {
    let p = dir("rng");
    let mut t = Trial::create(
        &p,
        "rng",
        87,
        "rng.v1",
        RandomWorld { total: 0 },
        RandomPolicy,
        Arc::new(ScriptedBackend),
        vec![AgentId(0), AgentId(1)],
        Scheduling::Random,
    )
    .unwrap();
    t.run_attempts(10).await.unwrap();
    t.checkpoint().unwrap();
    let expected = t.store.world().total;
    drop(t);
    let verified = Store::<RandomWorld, u64>::inspect_read_only(&p).unwrap();
    assert_eq!(verified.world.total, expected);
    let mut restored =
        Trial::resume(&p, "rng.v1", RandomPolicy, Arc::new(ScriptedBackend), false).unwrap();
    restored.run_attempts(2).await.unwrap();
    restored.checkpoint().unwrap();
    drop(restored);
    assert_eq!(
        Store::<RandomWorld, u64>::inspect_read_only(&p)
            .unwrap()
            .sequence,
        12
    );
    fs::remove_dir_all(p).unwrap();
}
struct RoundPolicy;
impl EffectPolicy<Society> for RoundPolicy {
    fn requests(&self, _: AgentId, _: &u64) -> Result<Vec<hiyal::runtime::EffectSpec>, String> {
        Ok(vec![sample_request("round")])
    }
    fn decide(
        &self,
        _: AgentId,
        obs: &u64,
        records: &[hiyal::runtime::Receipt],
    ) -> Result<u64, String> {
        if *obs != 1 || !matches!(records[0].result, EffectResult::Model { .. }) {
            return Err("round actors did not observe frozen world".into());
        }
        Ok(1)
    }
}
struct SlowBackend {
    active: AtomicUsize,
    max: AtomicUsize,
}
impl Backend for SlowBackend {
    fn resolve<'a>(&'a self, _: &'a hiyal::runtime::EffectSpec) -> BoxResult<'a> {
        Box::pin(async move {
            let n = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.max.fetch_max(n, Ordering::SeqCst);
            tokio::time::sleep(std::time::Duration::from_millis(40)).await;
            self.active.fetch_sub(1, Ordering::SeqCst);
            Ok(EffectResult::Model {
                content: "ok".into(),
                effective_model: "offline".into(),
                usage: None,
            })
        })
    }
}
#[tokio::test]
async fn synchronous_round_overlaps_effects_and_commits_in_order() {
    let p = dir("round");
    let backend = Arc::new(SlowBackend {
        active: AtomicUsize::new(0),
        max: AtomicUsize::new(0),
    });
    let mut trial = Trial::create(
        &p,
        "round",
        43,
        "society.round",
        Society {
            balances: vec![1; 2],
            transcript: vec![],
        },
        RoundPolicy,
        backend.clone(),
        vec![AgentId(0), AgentId(1)],
        Scheduling::RoundRobin,
    )
    .unwrap();
    let results = trial.run_round(2).await.unwrap();
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|e| matches!(e, Step::Accepted(_))));
    assert!(
        backend.max.load(Ordering::SeqCst) >= 2,
        "effects were serialized"
    );
    assert_eq!(trial.store.world().balances, [2, 2]);
    trial.checkpoint().unwrap();
    drop(trial);
    assert_eq!(
        Store::<Society, u64>::inspect_read_only(&p)
            .unwrap()
            .sequence,
        2
    );
    fs::remove_dir_all(p).unwrap();
}

struct OtherActorsStream;
impl EffectPolicy<RandomWorld> for OtherActorsStream {
    fn decide(&self, _: AgentId, _: &(), _: &[hiyal::runtime::Receipt]) -> Result<u64, String> {
        Ok(0)
    }
    fn decide_with_streams(
        &self,
        _: AgentId,
        _: &(),
        _: &[hiyal::runtime::Receipt],
        rng: &mut Streams,
    ) -> Result<u64, String> {
        rng.draw_u64("agent/1")
    }
}
#[tokio::test]
async fn policy_cannot_consume_another_agents_named_stream() {
    let p = dir("rng-scope");
    let mut trial = Trial::create(
        &p,
        "scope",
        81,
        "rng.scope",
        RandomWorld { total: 0 },
        OtherActorsStream,
        Arc::new(ScriptedBackend),
        vec![AgentId(0), AgentId(1)],
        Scheduling::RoundRobin,
    )
    .unwrap();
    assert!(
        trial
            .step()
            .await
            .err().expect("unauthorized draw must fail")
            .contains("unauthorized RNG stream")
    );
    assert_eq!(trial.store.sequence(), 0);
    drop(trial);
    fs::remove_dir_all(p).unwrap();
}
struct WrongWorldStream {
    value: u64,
}
impl Clone for WrongWorldStream {
    fn clone(&self) -> Self {
        Self { value: self.value }
    }
}
impl Serialize for WrongWorldStream {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.value.serialize(s)
    }
}
impl<'de> Deserialize<'de> for WrongWorldStream {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Ok(Self {
            value: u64::deserialize(d)?,
        })
    }
}
impl World for WrongWorldStream {
    type Action = u64;
    type Observation = ();
    fn observe(&self, _: AgentId) -> Result<(), String> {
        Ok(())
    }
    fn apply(&mut self, _: AgentId, action: u64) -> Result<(), String> {
        self.value += action;
        Ok(())
    }
    fn apply_with_streams(
        &mut self,
        a: AgentId,
        action: u64,
        rng: &mut Streams,
    ) -> Result<(), String> {
        rng.draw_u64("agent/1")?;
        self.apply(a, action)
    }
}
struct InnocentPolicy;
impl EffectPolicy<WrongWorldStream> for InnocentPolicy {
    fn decide(&self, _: AgentId, _: &(), _: &[hiyal::runtime::Receipt]) -> Result<u64, String> {
        Ok(1)
    }
}
#[tokio::test]
async fn reducer_cannot_consume_agents_stream() {
    let p = dir("world-scope");
    let mut trial = Trial::create(
        &p,
        "world-scope",
        81,
        "rng.world-scope",
        WrongWorldStream { value: 0 },
        InnocentPolicy,
        Arc::new(ScriptedBackend),
        vec![AgentId(0)],
        Scheduling::RoundRobin,
    )
    .unwrap();
    assert!(
        trial
            .step()
            .await
            .err().expect("unauthorized draw must fail")
            .contains("unauthorized RNG stream")
    );
    assert_eq!(trial.store.sequence(), 0);
    drop(trial);
    fs::remove_dir_all(p).unwrap();
}
#[tokio::test]
async fn trusted_projection_recovers_only_missing_suffix() {
    use hiyal::observe::{EventSummary, JsonlObserver};
    let p = dir("trust-projection");
    let mut trial = create_society(&p);
    trial.observe_with(Arc::new(JsonlObserver::recover(&p).unwrap()));
    trial.run_attempts(4).await.unwrap();
    trial.checkpoint().unwrap();
    drop(trial);
    let path = JsonlObserver::<Society>::path(&p);
    let all = fs::read_to_string(&path).unwrap();
    let first_three = all
        .lines()
        .take(3)
        .map(|s| format!("{s}\n"))
        .collect::<String>();
    assert_eq!(first_three.lines().count(), 3);
    let last: EventSummary = serde_json::from_str(first_three.lines().last().unwrap()).unwrap();
    assert!(last.source_log_bytes.is_some());
    fs::write(&path, first_three).unwrap();
    let resumed = Trial::resume_trusted(
        &p,
        "society.v1",
        Rules(Gain),
        Arc::new(ScriptedBackend),
        false,
    )
    .unwrap();
    let observer = JsonlObserver::recover_trusted(&resumed.store).unwrap();
    drop(observer);
    drop(resumed);
    let reconstructed = fs::read_to_string(&path).unwrap();
    assert_eq!(reconstructed.lines().count(), 4);
    assert_eq!(reconstructed, all);
    fs::remove_dir_all(p).unwrap();
}
