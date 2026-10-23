//! Read-only replay memo: cached views are detached from caller-held values,
//! and staged appends still fail fresh replay when a reducer is inconsistent.
use hiyal::{
    core::{AgentId, Policy, World},
    experiment::{Rules, Trial},
    runtime::ScriptedBackend,
    scheduler::Scheduling,
    storage::{Store, clear_replay_cache},
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{
    cell::RefCell,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

fn dir(tag: &str) -> std::path::PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    std::env::temp_dir().join(format!(
        "hiyal-replay-cache-{tag}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}

/// Legal but aliasing world state: `Clone` shares the cell instead of copying it.
#[derive(Clone)]
struct Shared(Rc<RefCell<u64>>);
impl Serialize for Shared {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.0.borrow().serialize(s)
    }
}
impl<'de> Deserialize<'de> for Shared {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Ok(Self(Rc::new(RefCell::new(u64::deserialize(d)?))))
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct Aliased {
    total: u64,
    shared: Shared,
}
impl World for Aliased {
    type Action = u64;
    type Observation = u64;
    fn observe(&self, _: AgentId) -> Result<u64, String> {
        Ok(self.total)
    }
    fn apply(&mut self, _: AgentId, amount: u64) -> Result<(), String> {
        self.total += amount;
        *self.shared.0.borrow_mut() = self.total;
        Ok(())
    }
}

struct One;
impl Policy<Aliased> for One {
    fn propose(&self, _: AgentId, _: &u64) -> Result<u64, String> {
        Ok(1)
    }
}

#[tokio::test]
async fn caller_mutation_of_a_cached_view_does_not_leak_into_later_hits() {
    let p = dir("alias");
    let mut trial = Trial::create(
        &p,
        "alias",
        3,
        "alias.v1",
        Aliased {
            total: 0,
            shared: Shared(Rc::new(RefCell::new(0))),
        },
        Rules(One),
        Arc::new(ScriptedBackend),
        vec![AgentId(0)],
        Scheduling::RoundRobin,
    )
    .unwrap();
    trial.run_attempts(4).await.unwrap();
    drop(trial);
    clear_replay_cache();
    let first = Store::<Aliased, u64>::inspect_read_only_cached(&p).unwrap();
    assert_eq!(*first.world.shared.0.borrow(), 4);
    *first.world.shared.0.borrow_mut() = 999;
    *first.genesis.initial_world.shared.0.borrow_mut() = 999;
    // A hit must equal a fresh uncached inspection, not the caller's edits.
    let second = Store::<Aliased, u64>::inspect_read_only_cached(&p).unwrap();
    let fresh = Store::<Aliased, u64>::inspect_read_only(&p).unwrap();
    assert_eq!(*second.world.shared.0.borrow(), 4);
    assert_eq!(
        *second.genesis.initial_world.shared.0.borrow(),
        *fresh.genesis.initial_world.shared.0.borrow()
    );
    std::fs::remove_dir_all(p).unwrap();
}

/// Reducer whose result depends on process state outside (world, action).
static DRIFT: AtomicU64 = AtomicU64::new(1);
#[derive(Clone, Serialize, Deserialize)]
struct Drifting {
    total: u64,
}
impl World for Drifting {
    type Action = u64;
    type Observation = u64;
    fn observe(&self, _: AgentId) -> Result<u64, String> {
        Ok(self.total)
    }
    fn apply(&mut self, _: AgentId, amount: u64) -> Result<(), String> {
        self.total += amount * DRIFT.load(Ordering::SeqCst);
        Ok(())
    }
}
struct Two;
impl Policy<Drifting> for Two {
    fn propose(&self, _: AgentId, _: &u64) -> Result<u64, String> {
        Ok(2)
    }
}

#[tokio::test]
async fn staged_commit_of_an_inconsistent_reducer_fails_fresh_replay() {
    let p = dir("drift");
    let mut trial = Trial::create(
        &p,
        "drift",
        4,
        "drift.v1",
        Drifting { total: 0 },
        Rules(Two),
        Arc::new(ScriptedBackend),
        vec![AgentId(0)],
        Scheduling::RoundRobin,
    )
    .unwrap();
    // Staged appends commit without re-running the reducer.
    assert_eq!(trial.run_attempts(3).await.unwrap(), 3);
    drop(trial);
    DRIFT.store(5, Ordering::SeqCst);
    clear_replay_cache();
    let fresh = Store::<Drifting, u64>::inspect_read_only(&p).err().unwrap();
    assert!(fresh.contains("replayed world state differs"), "{fresh}");
    assert!(Store::<Drifting, u64>::inspect_read_only_cached(&p).is_err());
    std::fs::remove_dir_all(p).unwrap();
}
