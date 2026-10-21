//! Randomized initial worlds drawn from a trial's own setup streams are
//! recorded in genesis, reproducible, confined to the setup family and
//! preserved across resume and paired forks.
use hiyal::{
    core::{AgentId, Policy, TrialId, World},
    experiment::{Rules, Trial, verify_setup},
    research::{paired_forks, verify_fork},
    rng::Streams,
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

fn dir(tag: &str) -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    std::env::temp_dir().join(format!(
        "hiyal-setup-{}-{}-{}",
        tag,
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Labeled {
    balances: Vec<u64>,
    boosted: Vec<bool>,
    links: Vec<usize>,
}
impl World for Labeled {
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
impl Policy<Labeled> for Gain {
    fn propose(&self, _: AgentId, _: &u64) -> Result<u64, String> {
        Ok(1)
    }
}

const AGENTS: usize = 8;

/// Balanced random labels by Fisher-Yates on `setup/labels`, and one random
/// out-link per agent on `setup/graph`.
fn setup(streams: &mut Streams) -> Result<Labeled, String> {
    let mut order: Vec<usize> = (0..AGENTS).collect();
    for i in (1..AGENTS).rev() {
        let j = streams.uniform_index("setup/labels", i + 1)?;
        order.swap(i, j);
    }
    let mut boosted = vec![false; AGENTS];
    for &i in &order[..AGENTS / 2] {
        boosted[i] = true;
    }
    let links = (0..AGENTS)
        .map(|i| {
            let k = streams.uniform_index("setup/graph", AGENTS - 1)?;
            Ok(if k >= i { k + 1 } else { k })
        })
        .collect::<Result<_, String>>()?;
    Ok(Labeled {
        balances: vec![10; AGENTS],
        boosted,
        links,
    })
}

fn create(p: &std::path::Path) -> Trial<Labeled, Rules<Gain>> {
    Trial::create_with_setup(
        p,
        "labeled",
        41,
        "labeled.v1",
        setup,
        Rules(Gain),
        Arc::new(ScriptedBackend),
        (0..AGENTS as u64).map(AgentId).collect(),
        Scheduling::Random,
    )
    .unwrap()
}

#[tokio::test]
async fn setup_draws_are_recorded_in_genesis_and_reproducible() {
    let (a, b) = (dir("record-a"), dir("record-b"));
    let mut trial = create(&a);
    let genesis = trial.store.genesis().clone();
    let names: Vec<_> = genesis.initial_streams.positions.keys().cloned().collect();
    assert_eq!(names, ["setup/graph", "setup/labels"]);

    // The same closure on the trial's fresh streams gives the same world.
    let mut fresh = Streams::new(TrialId::new("labeled", 41).0, 41);
    assert_eq!(genesis.initial_world, setup(&mut fresh).unwrap());
    assert_eq!(genesis.initial_streams, fresh);
    assert_eq!(
        genesis.initial_world.boosted.iter().filter(|b| **b).count(),
        AGENTS / 2
    );

    // Same name and seed in another directory: identical genesis state.
    let other = create(&b);
    assert_eq!(other.store.genesis().initial_world, genesis.initial_world);
    assert_eq!(
        other.store.genesis().initial_streams,
        genesis.initial_streams
    );
    drop(other);

    // Running, checkpointing and resuming leave the recorded setup intact.
    assert_eq!(trial.run_attempts(12).await.unwrap(), 12);
    trial.checkpoint().unwrap();
    drop(trial);
    let resumed: Trial<Labeled, Rules<Gain>> = Trial::resume(
        &a,
        "labeled.v1",
        Rules(Gain),
        Arc::new(ScriptedBackend),
        false,
    )
    .unwrap();
    assert_eq!(resumed.store.logical_step(), 12);
    assert_eq!(
        resumed.store.world().balances.iter().sum::<u64>(),
        10 * AGENTS as u64 + 12
    );
    drop(resumed);
    Store::<Labeled, u64>::inspect_read_only(&a).unwrap();
    verify_setup(&a, setup).unwrap();
    std::fs::remove_dir_all(a).unwrap();
    std::fs::remove_dir_all(b).unwrap();
}

#[test]
fn setup_may_not_touch_runtime_streams() {
    for (stream, expected) in [
        ("scheduler", "scheduler"),
        ("agent/0/opportunity", "unauthorized RNG stream draw"),
        ("world", "unauthorized RNG stream draw"),
    ] {
        let p = dir("scope");
        let result: Result<Trial<Labeled, Rules<Gain>>, String> = Trial::create_with_setup(
            &p,
            "labeled",
            41,
            "labeled.v1",
            |streams: &mut Streams| {
                streams.draw_u64(stream)?;
                setup(streams)
            },
            Rules(Gain),
            Arc::new(ScriptedBackend),
            (0..AGENTS as u64).map(AgentId).collect(),
            Scheduling::Random,
        );
        let error = result.err().unwrap();
        assert!(error.contains(expected), "{stream}: {error}");
        assert!(!p.exists(), "a rejected setup must not create a trial");
    }
}

#[test]
fn verify_setup_rejects_a_different_setup() {
    let p = dir("verify");
    drop(create(&p));
    // One extra draw changes the recorded stream position.
    let extra = verify_setup(&p, |streams: &mut Streams| {
        streams.draw_u64("setup/labels")?;
        setup(streams)
    });
    assert!(extra.is_err());
    // A different world from the same draws is caught too.
    let altered = verify_setup(&p, |streams: &mut Streams| {
        let mut world = setup(streams)?;
        world.balances[0] += 1;
        Ok(world)
    });
    assert!(altered.is_err());
    verify_setup(&p, setup).unwrap();
    std::fs::remove_dir_all(p).unwrap();
}

#[test]
fn paired_forks_carry_setup_positions() {
    let root = dir("forks");
    let parent = root.join("parent");
    let recorded = create(&parent).store.genesis().initial_streams.clone();
    let (control, treatment) = paired_forks(
        &parent,
        root.join("arms"),
        "boost-off",
        serde_json::json!({}),
        || Rules(Gain),
        |world: &mut Labeled| {
            world.boosted = vec![false; AGENTS];
            Ok(())
        },
        Arc::new(ScriptedBackend),
    )
    .unwrap();
    for arm in [&control, &treatment] {
        assert_eq!(
            arm.store.genesis().initial_streams.positions,
            recorded.positions
        );
    }
    drop((control, treatment));
    verify_fork::<Labeled>(root.join("arms/control")).unwrap();
    verify_fork::<Labeled>(root.join("arms/treatment")).unwrap();
    // Setup is audited on the trial that ran it, not on a fork.
    assert!(verify_setup(root.join("arms/control"), setup).is_err());
    verify_setup(&parent, setup).unwrap();
    std::fs::remove_dir_all(root).unwrap();
}
