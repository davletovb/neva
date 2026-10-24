//! Sharing verified replays: a fork's ancestry copy is served by its
//! checkpointed parent's memoized view, and `verify_fork_views` returns the
//! views `verify_fork` verified.
use hiyal::{
    core::{AgentId, Policy, World},
    experiment::{Rules, Trial},
    research::{fork_at_head, verify_fork, verify_fork_views},
    runtime::ScriptedBackend,
    scheduler::Scheduling,
    storage::{ReadOnly, Store, clear_replay_cache},
};
use serde::{Deserialize, Serialize};
use std::{
    cell::Cell,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

thread_local! {
    /// Reducer calls on this thread: a replay re-applies every event.
    static APPLIED: Cell<u64> = const { Cell::new(0) };
}
fn applied() -> u64 {
    APPLIED.with(Cell::get)
}

#[derive(Clone, Serialize, Deserialize)]
struct Tally {
    total: u64,
}
impl World for Tally {
    type Action = u64;
    type Observation = u64;
    fn observe(&self, _: AgentId) -> Result<u64, String> {
        Ok(self.total)
    }
    fn apply(&mut self, _: AgentId, amount: u64) -> Result<(), String> {
        APPLIED.with(|a| a.set(a.get() + 1));
        self.total += amount;
        Ok(())
    }
}
struct Add;
impl Policy<Tally> for Add {
    fn propose(&self, _: AgentId, _: &u64) -> Result<u64, String> {
        Ok(2)
    }
}

fn dir(tag: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    std::env::temp_dir().join(format!(
        "hiyal-fork-views-{tag}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}

/// A parent with 6 events and a checkpoint (so a `snapshot.json`).
async fn checkpointed_parent(path: &Path) {
    let mut trial = Trial::create(
        path,
        "views",
        11,
        "tally.v1",
        Tally { total: 0 },
        Rules(Add),
        Arc::new(ScriptedBackend),
        vec![AgentId(0), AgentId(1)],
        Scheduling::RoundRobin,
    )
    .unwrap();
    trial.run_attempts(6).await.unwrap();
    trial.checkpoint().unwrap();
}

/// The parent's manifest and events, without its snapshot.
fn snapshot_free_copy(parent: &Path, copy: &Path) {
    std::fs::create_dir_all(copy).unwrap();
    for name in ["manifest.json", "events.jsonl"] {
        std::fs::copy(parent.join(name), copy.join(name)).unwrap();
    }
}

fn same(a: &ReadOnly<Tally>, b: &ReadOnly<Tally>) {
    assert_eq!(
        serde_json::to_string(&(&a.world, &a.streams, &a.schedule, &a.genesis)).unwrap(),
        serde_json::to_string(&(&b.world, &b.streams, &b.schedule, &b.genesis)).unwrap()
    );
    assert_eq!(
        (a.sequence, a.logical_step, &a.chain_head, a.log_bytes),
        (b.sequence, b.logical_step, &b.chain_head, b.log_bytes)
    );
}

#[tokio::test]
async fn a_snapshot_free_copy_reuses_its_checkpointed_parents_verified_view() {
    let root = dir("reuse");
    let parent = root.join("parent");
    let copy = root.join("copy");
    checkpointed_parent(&parent).await;
    snapshot_free_copy(&parent, &copy);
    clear_replay_cache();
    Store::<Tally, u64>::inspect_read_only_cached(&parent).unwrap();
    let before = applied();
    let reused = Store::<Tally, u64>::inspect_read_only_cached(&copy).unwrap();
    assert_eq!(
        applied(),
        before,
        "served from the parent's view, not replayed"
    );
    same(
        &reused,
        &Store::<Tally, u64>::inspect_read_only(&copy).unwrap(),
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn a_snapshot_is_never_served_by_a_snapshot_free_view() {
    let root = dir("one-way");
    let parent = root.join("parent");
    let copy = root.join("copy");
    checkpointed_parent(&parent).await;
    snapshot_free_copy(&parent, &copy);
    clear_replay_cache();
    Store::<Tally, u64>::inspect_read_only_cached(&copy).unwrap();
    // The parent's snapshot must still be checked: it replays...
    let before = applied();
    Store::<Tally, u64>::inspect_read_only_cached(&parent).unwrap();
    assert!(
        applied() > before,
        "a snapshot is checked by its own replay"
    );
    // ...and a tampered snapshot fails, whatever the memo holds.
    clear_replay_cache();
    Store::<Tally, u64>::inspect_read_only_cached(&copy).unwrap();
    let path = parent.join("snapshot.json");
    let mut snapshot: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    snapshot["world"]["total"] = serde_json::json!(999);
    std::fs::write(&path, serde_json::to_vec(&snapshot).unwrap()).unwrap();
    let Err(error) = Store::<Tally, u64>::inspect_read_only_cached(&parent) else {
        panic!("a tampered snapshot must fail inspection");
    };
    assert!(error.contains("snapshot failed"), "{error}");
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn verify_fork_views_returns_what_fresh_inspections_give() {
    let root = dir("views");
    let parent = root.join("parent");
    let branch = root.join("branch");
    checkpointed_parent(&parent).await;
    let mut trial = fork_at_head(
        &parent,
        &branch,
        "test",
        "treatment",
        serde_json::json!({"bonus": 5}),
        |world: &mut Tally| {
            world.total += 5;
            Ok(())
        },
        Rules(Add),
        Arc::new(ScriptedBackend),
    )
    .unwrap();
    trial.run_attempts(4).await.unwrap();
    drop(trial);
    clear_replay_cache();
    let fork = verify_fork_views::<Tally>(&branch).unwrap();
    same(
        &fork.branch,
        &Store::<Tally, u64>::inspect_read_only(&branch).unwrap(),
    );
    same(
        &fork.parent,
        &Store::<Tally, u64>::inspect_read_only(branch.join("ancestry")).unwrap(),
    );
    assert_eq!(
        serde_json::to_string(&fork.metadata).unwrap(),
        serde_json::to_string(&verify_fork::<Tally>(&branch).unwrap()).unwrap()
    );
    assert_eq!(fork.branch.world.total, 12 + 5 + 8);
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn a_warm_memo_never_hides_tampering_with_a_verified_fork() {
    let root = dir("tamper");
    let parent = root.join("parent");
    let branch = root.join("branch");
    checkpointed_parent(&parent).await;
    let mut trial = fork_at_head(
        &parent,
        &branch,
        "test",
        "treatment",
        serde_json::json!({"bonus": 5}),
        |world: &mut Tally| {
            world.total += 5;
            Ok(())
        },
        Rules(Add),
        Arc::new(ScriptedBackend),
    )
    .unwrap();
    trial.run_attempts(4).await.unwrap();
    drop(trial);
    clear_replay_cache();
    // Warm the memo with every view of this fork and its parent.
    Store::<Tally, u64>::inspect_read_only_cached(&parent).unwrap();
    verify_fork_views::<Tally>(&branch).unwrap();

    // Each edit changes bytes the memo verified; each must fail anew.
    for file in [
        branch.join("events.jsonl"),
        branch.join("ancestry").join("events.jsonl"),
        branch.join("manifest.json"),
    ] {
        let original = std::fs::read(&file).unwrap();
        let text = String::from_utf8(original.clone()).unwrap();
        // One committed amount (2 becomes 3), or the initial total (a 1
        // put in front of its digits).
        let tampered = text.replacen("\"action\":2", "\"action\":3", 1).replacen(
            "\"total\":",
            "\"total\":1",
            1,
        );
        assert_ne!(tampered, text, "the edit must change {}", file.display());
        std::fs::write(&file, tampered).unwrap();
        assert!(
            verify_fork_views::<Tally>(&branch).is_err(),
            "tampering with {} must fail despite a warm memo",
            file.display()
        );
        std::fs::write(&file, original).unwrap();
        verify_fork_views::<Tally>(&branch).unwrap();
    }
    std::fs::remove_dir_all(root).unwrap();
}
