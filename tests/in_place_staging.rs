//! In-place staging (`World::apply_undoable`), in-place replay, and verified
//! trajectories from the journal (`Store::inspect_read_only_with`).
use hiyal::{
    core::{AgentId, Policy, Undo, World},
    experiment::{Rules, Trial},
    rng::Streams,
    runtime::ScriptedBackend,
    scheduler::Scheduling,
    storage::{MAX_EVENT_RECORD_BYTES, Outcome, Store},
};
use serde::{Deserialize, Serialize};
use std::{
    cell::Cell,
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

fn dir(tag: &str) -> std::path::PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    std::env::temp_dir().join(format!(
        "hiyal-in-place-{tag}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}

thread_local! {
    static CLONES: Cell<u64> = const { Cell::new(0) };
    static UNDOS: Cell<u64> = const { Cell::new(0) };
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
enum Act {
    /// Add to a counter and log the amount.
    Add(String, u64),
    /// Logged like `Add`, but too large to record as an event.
    Big(String),
    /// Refused before any change.
    Refuse,
}

/// A world with a cheap, exact undo. `faulty_undo` leaves the log entry
/// behind, to show a full replay catches a wrong undo.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Ledger {
    counts: BTreeMap<String, u64>,
    log: Vec<u64>,
    faulty_undo: bool,
}
impl Clone for Ledger {
    fn clone(&self) -> Self {
        CLONES.with(|c| c.set(c.get() + 1));
        Self {
            counts: self.counts.clone(),
            log: self.log.clone(),
            faulty_undo: self.faulty_undo,
        }
    }
}
impl Ledger {
    fn new(faulty_undo: bool) -> Self {
        Self {
            counts: BTreeMap::new(),
            log: Vec::new(),
            faulty_undo,
        }
    }
}
impl World for Ledger {
    type Action = Act;
    type Observation = u64;
    fn observe(&self, _: AgentId) -> Result<u64, String> {
        Ok(self.log.len() as u64)
    }
    fn apply(&mut self, actor: AgentId, action: Act) -> Result<(), String> {
        self.apply_undoable(actor, action, &mut Streams::new("unused".into(), 0))
            .map(drop)
    }
    fn apply_undoable(
        &mut self,
        _: AgentId,
        action: Act,
        _: &mut Streams,
    ) -> Result<Undo<Self>, String> {
        let (key, amount) = match action {
            Act::Add(key, amount) => (key, amount),
            Act::Big(text) => ("big".to_owned(), text.len() as u64),
            Act::Refuse => return Err("refused".into()),
        };
        let previous = self.counts.get(&key).copied();
        *self.counts.entry(key.clone()).or_insert(0) += amount;
        self.log.push(amount);
        let faulty = self.faulty_undo;
        Ok(Undo::revert(move |world: &mut Ledger| {
            UNDOS.with(|u| u.set(u.get() + 1));
            match previous {
                Some(value) => world.counts.insert(key, value),
                None => world.counts.remove(&key),
            };
            if !faulty {
                world.log.pop();
            }
        }))
    }
}

/// Adds on every turn, except one oversized action from agent 0 at log
/// length 2 and a refusal at log length 4.
#[derive(Default)]
struct Script {
    sent_big: AtomicBool,
}
impl Policy<Ledger> for Script {
    fn propose(&self, actor: AgentId, logged: &u64) -> Result<Act, String> {
        Ok(match logged {
            2 if actor.0 == 0 && !self.sent_big.swap(true, Ordering::Relaxed) => {
                Act::Big("X".repeat(MAX_EVENT_RECORD_BYTES))
            }
            4 => Act::Refuse,
            n => Act::Add(format!("k{}", n % 3), n + 1),
        })
    }
}

fn ledger_trial(
    path: &std::path::Path,
    faulty_undo: bool,
    agents: u64,
) -> Trial<Ledger, Rules<Script>> {
    Trial::create(
        path,
        "ledger",
        7,
        "ledger.v1",
        Ledger::new(faulty_undo),
        Rules(Script::default()),
        Arc::new(ScriptedBackend),
        (0..agents).map(AgentId).collect(),
        Scheduling::RoundRobin,
    )
    .unwrap()
}

fn json<T: Serialize>(value: &T) -> String {
    serde_json::to_string(value).unwrap()
}

#[tokio::test]
async fn a_world_with_its_own_undo_is_staged_without_cloning() {
    let p = dir("no-clone");
    let mut trial = ledger_trial(&p, false, 1);
    let before = CLONES.with(Cell::get);
    trial.run_attempts(12).await.unwrap();
    assert_eq!(
        CLONES.with(Cell::get),
        before,
        "staging, committing and rejecting must not clone this world"
    );
    let live = json(trial.store.world());
    drop(trial);
    let replayed = Store::<Ledger, Act>::inspect_read_only(&p).unwrap();
    assert_eq!(json(&replayed.world), live);
    std::fs::remove_dir_all(p).unwrap();
}

#[tokio::test]
async fn an_event_too_large_to_record_is_undone_and_recorded_as_rejected() {
    let p = dir("too-large");
    let mut trial = ledger_trial(&p, false, 1);
    let undos = UNDOS.with(Cell::get);
    trial.run_attempts(8).await.unwrap();
    assert_eq!(UNDOS.with(Cell::get), undos + 1, "one undo, for Big");
    assert!(!trial.store.world().counts.contains_key("big"));
    let live = json(trial.store.world());
    drop(trial);
    let mut outcomes = Vec::new();
    let replayed = Store::<Ledger, Act>::inspect_read_only_with(&p, |event, _| {
        outcomes.push(match &event.outcome {
            Outcome::Accepted { .. } => "accepted",
            Outcome::Rejected { .. } => "rejected",
            Outcome::Control { .. } => "control",
        });
        Ok(())
    })
    .unwrap();
    assert_eq!(json(&replayed.world), live);
    // Big, at log length 2, is rejected as too large; adds resume after it,
    // and the refusal at log length 4 is rejected too.
    assert_eq!(
        outcomes,
        [
            "accepted", "accepted", "rejected", "accepted", "accepted", "rejected", "rejected",
            "rejected"
        ]
    );
    std::fs::remove_dir_all(p).unwrap();
}

#[tokio::test]
async fn rounds_undo_an_oversized_action_and_keep_committing() {
    let p = dir("round");
    let mut trial = ledger_trial(&p, false, 3);
    for _ in 0..3 {
        trial.run_round(3).await.unwrap();
    }
    let live = json(trial.store.world());
    drop(trial);
    let replayed = Store::<Ledger, Act>::inspect_read_only(&p).unwrap();
    assert_eq!(json(&replayed.world), live);
    assert!(!replayed.world.counts.contains_key("big"));
    std::fs::remove_dir_all(p).unwrap();
}

#[tokio::test]
async fn a_wrong_undo_is_caught_by_full_replay() {
    let p = dir("faulty");
    let mut trial = ledger_trial(&p, true, 2);
    // Agent 0 meets Big at log length 2, its faulty undo leaves a log entry
    // behind, and agent 1 then commits a hash of that wrong world.
    trial.run_attempts(6).await.unwrap();
    drop(trial);
    let Err(error) = Store::<Ledger, Act>::inspect_read_only(&p) else {
        panic!("a wrong undo must fail full replay");
    };
    assert!(error.contains("differs from committed state"), "{error}");
    std::fs::remove_dir_all(p).unwrap();
}

/// A world whose reducer changes state before failing, with the default
/// `apply_undoable`.
#[derive(Clone, Serialize, Deserialize)]
struct HalfWay {
    total: u64,
}
impl World for HalfWay {
    type Action = u64;
    type Observation = u64;
    fn observe(&self, _: AgentId) -> Result<u64, String> {
        Ok(self.total)
    }
    fn apply(&mut self, _: AgentId, amount: u64) -> Result<(), String> {
        self.total += amount;
        if amount > 5 {
            return Err("too much, after changing the total".into());
        }
        Ok(())
    }
}
struct Ramp;
impl Policy<HalfWay> for Ramp {
    fn propose(&self, _: AgentId, total: &u64) -> Result<u64, String> {
        Ok(if *total == 3 { 9 } else { 1 })
    }
}

#[tokio::test]
async fn the_default_undo_restores_a_reducer_that_fails_halfway() {
    let p = dir("halfway");
    let mut trial = Trial::create(
        &p,
        "halfway",
        1,
        "halfway.v1",
        HalfWay { total: 0 },
        Rules(Ramp),
        Arc::new(ScriptedBackend),
        vec![AgentId(0)],
        Scheduling::RoundRobin,
    )
    .unwrap();
    trial.run_attempts(6).await.unwrap();
    // 1, 1, 1, then 9 is rejected (and keeps being proposed at total 3).
    assert_eq!(trial.store.world().total, 3);
    drop(trial);
    let replayed = Store::<HalfWay, u64>::inspect_read_only(&p).unwrap();
    assert_eq!(replayed.world.total, 3);
    std::fs::remove_dir_all(p).unwrap();
}

#[tokio::test]
async fn a_verified_trajectory_comes_from_the_journal_in_one_pass() {
    let p = dir("trajectory");
    let mut trial = ledger_trial(&p, false, 2);
    trial.run_attempts(10).await.unwrap();
    trial.checkpoint().unwrap();
    trial.run_attempts(4).await.unwrap();
    drop(trial);

    // The trajectory: the action and the world after every committed event.
    let mut sequences = Vec::new();
    let mut trajectory = Vec::new();
    let view = Store::<Ledger, Act>::inspect_read_only_with(&p, |event, after| {
        sequences.push(event.sequence());
        if let Outcome::Accepted { action, .. } = &event.outcome {
            trajectory.push((event.logical_step(), action.clone(), after.log.clone()));
        }
        Ok(())
    })
    .unwrap();
    let plain = Store::<Ledger, Act>::inspect_read_only(&p).unwrap();
    assert_eq!(json(&view.world), json(&plain.world));
    assert_eq!(sequences, (0..view.sequence).collect::<Vec<_>>());

    // An independent re-application gives the same trajectory.
    let mut world = Ledger::new(false);
    for (step, (logical_step, action, log)) in trajectory.iter().enumerate() {
        assert_eq!(*logical_step, step as u64);
        world.apply(AgentId(0), action.clone()).unwrap();
        assert_eq!(&world.log, log);
    }
    assert_eq!(json(&world), json(&plain.world));

    // A visitor's error stops the replay and is returned.
    let mut seen = 0;
    let Err(error) = Store::<Ledger, Act>::inspect_read_only_with(&p, |_, _| {
        seen += 1;
        if seen == 3 {
            Err("enough".into())
        } else {
            Ok(())
        }
    }) else {
        panic!("a visitor error must stop the replay");
    };
    assert_eq!(error, "enough");
    assert_eq!(seen, 3);
    std::fs::remove_dir_all(p).unwrap();
}
