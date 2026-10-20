//! Run worlds from external Rust crates; network effects never run in reducers.
use crate::{
    core::{AgentId, TrialId, World},
    rng::Streams,
    runtime::{Backend, EffectJournal, EffectSpec, Receipt},
    scheduler::{Schedule, Scheduling},
    storage::{BuildProvenance, Entry, Genesis, Outcome, SCHEMA, Store},
};
use serde::{Serialize, de::DeserializeOwned};
use std::{path::Path, sync::Arc, time::Duration};

pub trait EffectPolicy<W: World>: Send + Sync {
    fn requests(&self, _actor: AgentId, _obs: &W::Observation) -> Result<Vec<EffectSpec>, String> {
        Ok(vec![])
    }
    fn decide(
        &self,
        actor: AgentId,
        obs: &W::Observation,
        effects: &[Receipt],
    ) -> Result<W::Action, String>;
}
pub struct Rules<P>(pub P);
impl<W: World, P: crate::core::Policy<W> + Send + Sync> EffectPolicy<W> for Rules<P> {
    fn decide(&self, a: AgentId, o: &W::Observation, _: &[Receipt]) -> Result<W::Action, String> {
        self.0.propose(a, o)
    }
}
pub trait Observer<W: World>: Send + Sync {
    fn committed(&self, entry: &Entry<W, W::Action>) -> Result<(), String>;
}
#[derive(Debug)]
pub enum Step<W: World> {
    Accepted(Box<Entry<W, W::Action>>),
    Rejected { actor: AgentId, reason: String },
    NoEligible,
}
pub struct Trial<W, P>
where
    W: World + Serialize + DeserializeOwned,
    W::Action: Serialize + DeserializeOwned + Clone,
    P: EffectPolicy<W>,
{
    pub store: Store<W, W::Action>,
    pub effects: EffectJournal,
    pub policy: P,
    pub backend: Arc<dyn Backend>,
    pub replay_only: bool,
    pub concurrency: usize,
    pub timeout: Duration,
    observers: Vec<Arc<dyn Observer<W>>>,
    pub diagnostics: Vec<String>,
}
impl<W, P> Trial<W, P>
where
    W: World + Serialize + DeserializeOwned,
    W::Action: Serialize + DeserializeOwned + Clone,
    P: EffectPolicy<W>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        dir: impl AsRef<Path>,
        name: &str,
        seed: u64,
        kind: &str,
        world: W,
        policy: P,
        backend: Arc<dyn Backend>,
        agents: Vec<AgentId>,
        mode: Scheduling,
    ) -> Result<Self, String> {
        let trial = TrialId::new(name, seed).0;
        let genesis = Genesis {
            schema: SCHEMA,
            trial_id: trial.clone(),
            initial_world: world,
            initial_schedule: Schedule::new(mode, agents)?,
            initial_streams: Streams::new(trial, seed),
            world_kind: kind.into(),
            build: Some(BuildProvenance::current()),
        };
        let store = Store::create(&dir, genesis)?;
        let mut effects = EffectJournal::open(&dir)?;
        effects.bind_trial(&store.genesis.trial_id)?;
        backend.recover_admission(dir.as_ref(), effects.receipts())?;
        Ok(Self {
            store,
            effects,
            policy,
            backend,
            replay_only: false,
            concurrency: 4,
            timeout: Duration::from_secs(15),
            observers: vec![],
            diagnostics: vec![],
        })
    }
    /// Create a trial from an independently verified research fork genesis.
    /// The schedule/RNG state can begin at a parent's committed boundary.
    pub fn create_from_genesis(
        dir: impl AsRef<Path>,
        genesis: Genesis<W>,
        policy: P,
        backend: Arc<dyn Backend>,
    ) -> Result<Self, String> {
        let store = Store::create(&dir, genesis)?;
        let mut effects = EffectJournal::open(&dir)?;
        effects.bind_trial(&store.genesis.trial_id)?;
        backend.recover_admission(dir.as_ref(), effects.receipts())?;
        Ok(Self {
            store,
            effects,
            policy,
            backend,
            replay_only: false,
            concurrency: 4,
            timeout: Duration::from_secs(15),
            observers: vec![],
            diagnostics: vec![],
        })
    }

    pub fn resume(
        dir: impl AsRef<Path>,
        kind: &str,
        policy: P,
        backend: Arc<dyn Backend>,
        replay_only: bool,
    ) -> Result<Self, String> {
        let store = Store::open(&dir)?;
        if store.genesis.world_kind != kind {
            return Err("wrong registered world kind".into());
        }
        let mut effects = EffectJournal::open_existing(dir)?;
        effects.bind_trial(&store.genesis.trial_id)?;
        backend.recover_admission(dir.as_ref(), effects.receipts())?;
        Ok(Self {
            store,
            effects,
            policy,
            backend,
            replay_only,
            concurrency: 4,
            timeout: Duration::from_secs(15),
            observers: vec![],
            diagnostics: vec![],
        })
    }
    pub fn observe_with(&mut self, obs: Arc<dyn Observer<W>>) {
        self.observers.push(obs);
    }
    // All durable log entries (accepted, rejected, control) reach observers.
    // Observer errors are diagnostic and must never roll back a committed event.
    fn emit(&mut self, entry: &Entry<W, W::Action>) {
        for observer in &self.observers {
            if let Err(e) = observer.committed(entry) {
                self.diagnostics.push(format!("observer failed: {e}"));
            }
        }
    }
    /// Scheduler control is an audited event, never a raw in-memory mutation.
    pub fn pause_actor(&mut self, actor: AgentId) -> Result<(), String> {
        let entry = self.store.set_paused_event(actor, true)?;
        self.emit(&entry);
        Ok(())
    }
    pub fn resume_actor(&mut self, actor: AgentId) -> Result<(), String> {
        let entry = self.store.set_paused_event(actor, false)?;
        self.emit(&entry);
        Ok(())
    }
    pub async fn step(&mut self) -> Result<Step<W>, String> {
        let mut sched = self.store.schedule.clone();
        let mut rng = self.store.streams.clone();
        let Some(actor) = sched.select(&mut rng)? else {
            return Ok(Step::NoEligible);
        };
        // The attempt sequence, not only the accepted-world step, forms the
        // effect call key; failed attempts cannot reuse paid calls accidentally.
        let result = match self.store.world.observe(actor) {
            Ok(obs) => match self.policy.requests(actor, &obs) {
                Ok(specs) => {
                    // Reject oversized *new* policies as a committed attempt
                    // rejection, but permit complete historic ledger replays.
                    let over_limit = specs.len() > 32
                        || specs
                            .iter()
                            .filter(|s| matches!(s, crate::runtime::EffectSpec::Tool { .. }))
                            .count()
                            > 8;
                    let all_recorded = over_limit
                        && specs.iter().enumerate().all(|(i, spec)| {
                            let Ok(index) = u32::try_from(i) else {
                                return false;
                            };
                            let Ok(key) = spec.key(
                                &self.store.genesis.trial_id,
                                self.store.sequence,
                                actor,
                                index,
                            ) else {
                                return false;
                            };
                            self.effects
                                .receipts()
                                .get(&key)
                                .is_some_and(|r| r.spec == *spec)
                        });
                    if over_limit && !all_recorded {
                        Err("policy exceeded 32 effects or 8 tool calls; no external calls launched".into())
                    } else {
                        let mut keyed = Vec::new();
                        for (index, spec) in specs.into_iter().enumerate() {
                            let index =
                                u32::try_from(index).map_err(|_| "too many effect calls")?;
                            let key = spec.key(
                                &self.store.genesis.trial_id,
                                self.store.sequence,
                                actor,
                                index,
                            )?;
                            keyed.push((key, spec));
                        }
                        let receipts = self
                            .effects
                            .resolve_all(
                                keyed,
                                self.backend.clone(),
                                self.concurrency,
                                self.timeout,
                                self.replay_only,
                            )
                            .await?; // Storage, replay-integrity and task failures are fatal.
                        if receipts.iter().any(|r| {
                            matches!(
                                &r.result,
                                crate::effects::EffectResult::Failed { .. }
                                    | crate::effects::EffectResult::UnsupportedCapability { .. }
                            )
                        }) {
                            Err("effect failed or capability unsupported".into())
                        } else {
                            self.policy.decide(actor, &obs, &receipts)
                        }
                    }
                }
                Err(e) => Err(e),
            },
            Err(e) => Err(e),
        };
        match result {
            Ok(action) => {
                let mut staged = self.store.world.clone();
                if let Err(reason) = staged.apply(actor, action.clone()) {
                    let entry = self.store.append(
                        actor,
                        Outcome::Rejected {
                            reason: reason.clone(),
                        },
                        sched,
                        rng,
                    )?;
                    self.emit(&entry);
                    return Ok(Step::Rejected { actor, reason });
                }
                let state_hash = crate::storage::digest(&staged)?;
                let append = self.store.append(
                    actor,
                    Outcome::Accepted {
                        action,
                        world: None,
                        state_hash,
                    },
                    sched.clone(),
                    rng.clone(),
                );
                match append {
                    Ok(entry) => {
                        self.emit(&entry);
                        Ok(Step::Accepted(Box::new(entry)))
                    }
                    Err(error) if error == crate::storage::EVENT_TOO_LARGE => {
                        // Admission failed before any append. Preserve effects
                        // but record a rejected attempt so replay can progress.
                        let reason = error;
                        let entry = self.store.append(
                            actor,
                            Outcome::Rejected {
                                reason: reason.clone(),
                            },
                            sched,
                            rng,
                        )?;
                        self.emit(&entry);
                        Ok(Step::Rejected { actor, reason })
                    }
                    Err(error) => Err(error),
                }
            }
            Err(reason) => {
                let entry = self.store.append(
                    actor,
                    Outcome::Rejected {
                        reason: reason.clone(),
                    },
                    sched,
                    rng,
                )?;
                self.emit(&entry);
                Ok(Step::Rejected { actor, reason })
            }
        }
    }
    pub async fn run_attempts(&mut self, n: u64) -> Result<u64, String> {
        let mut accepted = 0;
        for _ in 0..n {
            match self.step().await? {
                Step::Accepted(_) => accepted += 1,
                Step::Rejected { .. } => {}
                Step::NoEligible => break,
            }
        }
        Ok(accepted)
    }
    pub fn checkpoint(&self) -> Result<(), String> {
        self.store.snapshot()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effects::EffectResult;
    use crate::runtime::{ReplayBackend, ScriptedBackend, sample_request};
    use serde::{Deserialize, Serialize};
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    fn dir(tag: &str) -> std::path::PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let p = std::env::temp_dir().join(format!("hiyal-{}-{tag}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        p
    }
    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct Counter {
        value: u64,
        reject: bool,
    }
    impl World for Counter {
        type Action = u64;
        type Observation = u64;
        fn observe(&self, _: AgentId) -> Result<u64, String> {
            Ok(self.value)
        }
        fn apply(&mut self, _: AgentId, delta: u64) -> Result<(), String> {
            if self.reject {
                return Err("rejected".into());
            }
            self.value = self.value.checked_add(delta).ok_or("overflow")?;
            Ok(())
        }
    }
    struct Policy {
        effects: bool,
    }
    impl EffectPolicy<Counter> for Policy {
        fn requests(&self, a: AgentId, _: &u64) -> Result<Vec<EffectSpec>, String> {
            if self.effects {
                Ok(vec![sample_request(&format!("agent-{}", a.0))])
            } else {
                Ok(vec![])
            }
        }
        fn decide(&self, _: AgentId, _: &u64, receipts: &[Receipt]) -> Result<u64, String> {
            if self.effects
                && !matches!(
                    receipts.first().map(|r| &r.result),
                    Some(EffectResult::Model { .. })
                )
            {
                return Err("missing model".into());
            }
            Ok(1)
        }
    }
    fn new(path: &Path, random: bool, effects: bool) -> Trial<Counter, Policy> {
        Trial::create(
            path,
            "society",
            57,
            "counter",
            Counter {
                value: 0,
                reject: false,
            },
            Policy { effects },
            Arc::new(ScriptedBackend),
            vec![AgentId(0), AgentId(1), AgentId(2)],
            if random {
                Scheduling::Random
            } else {
                Scheduling::RoundRobin
            },
        )
        .unwrap()
    }
    #[tokio::test]
    async fn resumed_random_trial_matches_uninterrupted_hashes() {
        let a = dir("whole");
        let b = dir("resume");
        let (expected, hash) = {
            let mut t = new(&a, true, false);
            assert_eq!(t.run_attempts(13).await.unwrap(), 13);
            t.checkpoint().unwrap();
            (
                t.store
                    .read_events()
                    .unwrap()
                    .into_iter()
                    .map(|e| e.hash)
                    .collect::<Vec<_>>(),
                t.store.chain_head.clone(),
            )
        };
        {
            let mut t = new(&b, true, false);
            t.run_attempts(5).await.unwrap();
            t.checkpoint().unwrap();
        }
        {
            let mut t = Trial::resume(
                &b,
                "counter",
                Policy { effects: false },
                Arc::new(ScriptedBackend),
                false,
            )
            .unwrap();
            t.run_attempts(8).await.unwrap();
            assert_eq!(t.store.chain_head, hash);
            assert_eq!(
                t.store
                    .read_events()
                    .unwrap()
                    .into_iter()
                    .map(|e| e.hash)
                    .collect::<Vec<_>>(),
                expected
            );
            t.checkpoint().unwrap();
        }
        std::fs::remove_dir_all(a).unwrap();
        std::fs::remove_dir_all(b).unwrap();
    }
    #[tokio::test]
    async fn malformed_snapshot_is_rejected_without_mutation() {
        let path = dir("corrupt");
        {
            let mut t = new(&path, false, false);
            t.run_attempts(4).await.unwrap();
            t.checkpoint().unwrap();
        }
        let file = path.join("snapshot.json");
        let mut json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&file).unwrap()).unwrap();
        json["sequence"] = serde_json::json!("777");
        let original = std::fs::read(&file).unwrap();
        std::fs::write(&file, serde_json::to_vec(&json).unwrap()).unwrap();
        assert!(
            Trial::resume(
                &path,
                "counter",
                Policy { effects: false },
                Arc::new(ScriptedBackend),
                false
            )
            .is_err()
        );
        assert!(path.join(".writer.lock").exists()); // kernel releases lock, not path
        std::fs::write(&file, original).unwrap();
        let valid = Trial::resume(
            &path,
            "counter",
            Policy { effects: false },
            Arc::new(ScriptedBackend),
            false,
        )
        .unwrap();
        assert_eq!(valid.store.sequence, 4);
        drop(valid);
        std::fs::remove_dir_all(path).unwrap();
    }
    #[tokio::test]
    async fn rejected_effect_keeps_receipt_and_world_unchanged() {
        let path = dir("reject");
        {
            let mut t = Trial::create(
                &path,
                "s",
                4,
                "counter",
                Counter {
                    value: 0,
                    reject: true,
                },
                Policy { effects: true },
                Arc::new(ScriptedBackend),
                vec![AgentId(0)],
                Scheduling::RoundRobin,
            )
            .unwrap();
            assert!(matches!(t.step().await.unwrap(), Step::Rejected { .. }));
            assert_eq!(t.store.world.value, 0);
            assert_eq!(t.store.sequence, 1);
            assert_eq!(t.store.logical_step, 0);
            assert_eq!(t.effects.receipts.len(), 1);
            t.checkpoint().unwrap();
        }
        {
            let t = Trial::resume(
                &path,
                "counter",
                Policy { effects: true },
                Arc::new(ScriptedBackend),
                true,
            )
            .unwrap();
            assert_eq!(t.store.world.value, 0);
            assert_eq!(t.effects.receipts.len(), 1);
        }
        std::fs::remove_dir_all(path).unwrap();
    }
    struct Failing;
    impl Observer<Counter> for Failing {
        fn committed(&self, _: &Entry<Counter, u64>) -> Result<(), String> {
            Err("unavailable".into())
        }
    }
    #[tokio::test]
    async fn observer_failure_does_not_mutate_committed_history() {
        let a = dir("obs1");
        let b = dir("obs2");
        let ha = {
            let mut t = new(&a, false, false);
            t.observe_with(Arc::new(Failing));
            t.run_attempts(3).await.unwrap();
            assert_eq!(t.diagnostics.len(), 3);
            t.store.chain_head.clone()
        };
        let hb = {
            let mut t = new(&b, false, false);
            t.run_attempts(3).await.unwrap();
            t.store.chain_head.clone()
        };
        assert_eq!(ha, hb);
        std::fs::remove_dir_all(a).unwrap();
        std::fs::remove_dir_all(b).unwrap();
    }
    #[tokio::test]
    async fn keyed_replay_is_independent_of_recorded_completion_order() {
        let path = dir("effects");
        std::fs::create_dir_all(&path).unwrap();
        let mut journal = EffectJournal::open(&path).unwrap();
        let a = sample_request("a");
        let b = sample_request("b");
        let ka = a.key("trial", 0, AgentId(0), 0).unwrap();
        let kb = b.key("trial", 0, AgentId(1), 0).unwrap();
        let recorded = journal
            .resolve_all(
                vec![(kb.clone(), b.clone()), (ka.clone(), a.clone())],
                Arc::new(ScriptedBackend),
                2,
                Duration::from_secs(2),
                false,
            )
            .await
            .unwrap();
        assert_eq!(recorded[0].key, ka);
        let replay = ReplayBackend::from_receipts(recorded.clone());
        assert_eq!(
            replay.resolve_key(&kb, &b).unwrap().result,
            recorded[1].result
        );
        assert!(replay.resolve_key(&kb, &a).is_err());
        let received = journal
            .resolve_all(
                vec![(ka, a), (kb, b)],
                Arc::new(ScriptedBackend),
                2,
                Duration::from_secs(2),
                true,
            )
            .await
            .unwrap();
        assert_eq!(recorded, received);
        drop(journal);
        std::fs::remove_dir_all(path).unwrap();
    }
    #[tokio::test]
    async fn persisted_pause_and_resume_survive_restart_without_divergence() {
        let path = dir("pause");
        {
            let mut trial = new(&path, false, false);
            trial.pause_actor(AgentId(0)).unwrap();
            let entry = trial.step().await.unwrap();
            assert!(matches!(entry, Step::Accepted(ref e) if e.actor == AgentId(1)));
            trial.checkpoint().unwrap();
        }
        {
            let mut trial = Trial::resume(
                &path,
                "counter",
                Policy { effects: false },
                Arc::new(ScriptedBackend),
                false,
            )
            .unwrap();
            let entry = trial.step().await.unwrap();
            assert!(matches!(entry, Step::Accepted(ref e) if e.actor == AgentId(2)));
            trial.resume_actor(AgentId(0)).unwrap();
            trial.checkpoint().unwrap();
        }
        let verified = Trial::resume(
            &path,
            "counter",
            Policy { effects: false },
            Arc::new(ScriptedBackend),
            false,
        )
        .unwrap();
        assert_eq!(verified.store.logical_step, 2);
        drop(verified);
        std::fs::remove_dir_all(path).unwrap();
    }
    #[tokio::test]
    async fn abandoned_snapshot_temp_is_cleaned_during_restore() {
        let path = dir("temporary");
        {
            let mut trial = new(&path, false, false);
            trial.run_attempts(2).await.unwrap();
            trial.checkpoint().unwrap();
        }
        std::fs::write(path.join("snapshot.tmp"), b"abandoned partial snapshot").unwrap();
        {
            let trial = Trial::resume(
                &path,
                "counter",
                Policy { effects: false },
                Arc::new(ScriptedBackend),
                false,
            )
            .unwrap();
            assert!(!path.join("snapshot.tmp").exists());
            trial.checkpoint().unwrap();
        }
        std::fs::remove_dir_all(path).unwrap();
    }
    struct Delayed;
    impl Backend for Delayed {
        fn resolve<'a>(&'a self, spec: &'a EffectSpec) -> crate::runtime::BoxResult<'a> {
            Box::pin(async move {
                let message = match spec {
                    EffectSpec::Model(model) => model.messages[0].content.as_str(),
                    _ => return Err("not a model".into()),
                };
                let delay = if message == "slow" { 70 } else { 1 };
                tokio::time::sleep(Duration::from_millis(delay)).await;
                Ok(EffectResult::Model {
                    content: message.into(),
                    effective_model: "delayed".into(),
                    usage: None,
                })
            })
        }
    }
    #[tokio::test]
    async fn out_of_order_effect_completions_have_stable_result_order() {
        let path = dir("out-of-order");
        std::fs::create_dir_all(&path).unwrap();
        let mut journal = EffectJournal::open(&path).unwrap();
        let slow = sample_request("slow");
        let fast = sample_request("fast");
        let slow_key = slow.key("trial", 0, AgentId(0), 0).unwrap();
        let fast_key = fast.key("trial", 0, AgentId(1), 0).unwrap();
        let results = journal
            .resolve_all(
                vec![(slow_key.clone(), slow), (fast_key.clone(), fast)],
                Arc::new(Delayed),
                2,
                Duration::from_secs(2),
                false,
            )
            .await
            .unwrap();
        assert_eq!(results[0].key, slow_key);
        assert_eq!(results[1].key, fast_key);
        let stored = std::fs::read_to_string(path.join("effects.jsonl")).unwrap();
        let first: Receipt = serde_json::from_str(stored.lines().next().unwrap()).unwrap();
        assert_eq!(first.key, fast_key);
        drop(journal);
        std::fs::remove_dir_all(path).unwrap();
    }
    struct NeverCallBackend;
    impl Backend for NeverCallBackend {
        fn resolve<'a>(&'a self, _: &'a EffectSpec) -> crate::runtime::BoxResult<'a> {
            Box::pin(async { panic!("recorded replay must not contact backend") })
        }
    }
    #[tokio::test]
    async fn receipt_saved_before_commit_replays_on_resumed_attempt() {
        let path = dir("interrupted");
        {
            let mut trial = new(&path, false, true);
            let request = sample_request("agent-0");
            let key = request
                .key(&trial.store.genesis.trial_id, 0, AgentId(0), 0)
                .unwrap();
            trial
                .effects
                .resolve_all(
                    vec![(key, request)],
                    Arc::new(ScriptedBackend),
                    1,
                    Duration::from_secs(2),
                    false,
                )
                .await
                .unwrap();
            assert_eq!(trial.store.sequence, 0);
        }
        {
            let mut resumed = Trial::resume(
                &path,
                "counter",
                Policy { effects: true },
                Arc::new(NeverCallBackend),
                true,
            )
            .unwrap();
            assert!(matches!(resumed.step().await.unwrap(), Step::Accepted(_)));
            assert_eq!(resumed.store.sequence, 1);
            assert_eq!(resumed.effects.receipts.len(), 1);
        }
        std::fs::remove_dir_all(path).unwrap();
    }
    #[tokio::test]
    async fn missing_replay_receipt_is_infrastructure_error_not_world_rejection() {
        let path = dir("strict-replay");
        let mut trial = new(&path, false, true);
        trial.replay_only = true;
        assert!(trial.step().await.is_err());
        assert_eq!(trial.store.sequence, 0);
        assert_eq!(trial.store.logical_step, 0);
        assert!(trial.store.read_events().unwrap().is_empty());
        drop(trial);
        std::fs::remove_dir_all(path).unwrap();
    }

    #[tokio::test]
    async fn foreign_effect_receipt_is_rejected_on_resume() {
        let one = dir("receipt-origin");
        let two = dir("receipt-destination");
        {
            let mut trial = new(&one, false, true);
            trial.run_attempts(1).await.unwrap();
        }
        {
            let _trial = Trial::create(
                &two,
                "other-name",
                57,
                "counter",
                Counter {
                    value: 0,
                    reject: false,
                },
                Policy { effects: true },
                Arc::new(ScriptedBackend),
                vec![AgentId(0), AgentId(1), AgentId(2)],
                Scheduling::RoundRobin,
            )
            .unwrap();
        }
        std::fs::copy(one.join("effects.jsonl"), two.join("effects.jsonl")).unwrap();
        assert!(
            Trial::resume(
                &two,
                "counter",
                Policy { effects: true },
                Arc::new(ScriptedBackend),
                true
            )
            .is_err()
        );
        assert!(two.join(".writer.lock").exists()); // marker inode is intentionally retained
        std::fs::remove_dir_all(one).unwrap();
        std::fs::remove_dir_all(two).unwrap();
    }

    #[tokio::test]
    async fn read_only_inspection_does_not_lock_or_modify_an_active_trial() {
        let path = dir("read-only");
        let mut trial = new(&path, false, false);
        trial.run_attempts(3).await.unwrap();
        let original = std::fs::read(path.join("events.jsonl")).unwrap();
        let view = Store::<Counter, u64>::inspect_read_only(&path).unwrap();
        assert_eq!(view.world.value, 3);
        assert_eq!(view.sequence, 3);
        assert_eq!(
            EffectJournal::inspect_read_only(&path, &trial.store.genesis.trial_id).unwrap(),
            0
        );
        assert_eq!(std::fs::read(path.join("events.jsonl")).unwrap(), original);
        assert!(path.join(".writer.lock").exists());
        drop(trial);
        assert!(path.join(".writer.lock").exists()); // kernel releases lock, not path
        std::fs::remove_dir_all(path).unwrap();
    }

    struct Slow;
    impl Backend for Slow {
        fn resolve<'a>(&'a self, _: &'a EffectSpec) -> crate::runtime::BoxResult<'a> {
            Box::pin(async {
                tokio::time::sleep(Duration::from_millis(80)).await;
                Ok(EffectResult::Model {
                    content: "late".into(),
                    effective_model: "late".into(),
                    usage: None,
                })
            })
        }
    }
    #[tokio::test]
    async fn timeout_records_failure_and_skips_world_commit() {
        let path = dir("timeout");
        let mut t = Trial::create(
            &path,
            "s",
            4,
            "counter",
            Counter {
                value: 0,
                reject: false,
            },
            Policy { effects: true },
            Arc::new(Slow),
            vec![AgentId(0)],
            Scheduling::RoundRobin,
        )
        .unwrap();
        t.timeout = Duration::from_millis(3);
        assert!(matches!(t.step().await.unwrap(), Step::Rejected { .. }));
        assert_eq!(t.store.logical_step, 0);
        assert_eq!(t.effects.receipts.len(), 1);
        assert!(matches!(
            t.effects.receipts.values().next().unwrap().result,
            EffectResult::Failed { .. }
        ));
        drop(t);
        std::fs::remove_dir_all(path).unwrap();
    }
}
