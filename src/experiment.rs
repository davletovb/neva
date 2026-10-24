//! Run worlds from external Rust crates; network effects never run in reducers.
use crate::{
    core::{AgentId, TrialId, World},
    rng::Streams,
    runtime::{Backend, EffectJournal, EffectSpec, Receipt},
    scheduler::{Schedule, Scheduling},
    storage::{AppendError, BuildProvenance, Entry, Genesis, Outcome, SCHEMA, Store},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
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
    fn requests_with_streams(
        &self,
        actor: AgentId,
        obs: &W::Observation,
        _rng: &mut Streams,
    ) -> Result<Vec<EffectSpec>, String> {
        self.requests(actor, obs)
    }
    fn decide_with_streams(
        &self,
        actor: AgentId,
        obs: &W::Observation,
        effects: &[Receipt],
        _rng: &mut Streams,
    ) -> Result<W::Action, String> {
        self.decide(actor, obs, effects)
    }
}
pub struct Rules<P>(pub P);
impl<W: World, P: crate::core::Policy<W> + Send + Sync> EffectPolicy<W> for Rules<P> {
    fn decide(&self, a: AgentId, o: &W::Observation, _: &[Receipt]) -> Result<W::Action, String> {
        self.0.propose(a, o)
    }
    fn decide_with_streams(
        &self,
        a: AgentId,
        o: &W::Observation,
        _: &[Receipt],
        rng: &mut Streams,
    ) -> Result<W::Action, String> {
        self.0.propose_with_streams(a, o, rng)
    }
}
pub trait Observer<W: World>: Send + Sync {
    fn committed(&self, entry: &Entry<W, W::Action>) -> Result<(), String>;
    /// The authoritative byte boundary enables safe suffix recovery of
    /// derived projections without changing existing observer implementations.
    fn committed_with_log_offset(
        &self,
        entry: &Entry<W, W::Action>,
        _end: u64,
    ) -> Result<(), String> {
        self.committed(entry)
    }
}
#[derive(Debug)]
pub enum Step<W: World> {
    Accepted(Box<Entry<W, W::Action>>),
    Rejected { actor: AgentId, reason: String },
    NoEligible,
}

const ROUND_PLAN_SCHEMA: u32 = 1;
const ROUND_PLAN_FILE: &str = "pending-round.json";
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PlannedActor {
    actor: AgentId,
    observation: Option<serde_json::Value>,
    reason: Option<String>,
    specs: Vec<(crate::effects::EffectCallKey, EffectSpec)>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RoundPlan {
    schema: u32,
    trial_id: String,
    start_sequence: u64,
    start_chain_head: String,
    actors: Vec<PlannedActor>,
    content_hash: String,
}
impl RoundPlan {
    fn hash(&self) -> Result<String, String> {
        let mut value = serde_json::to_value(self).map_err(|e| e.to_string())?;
        value
            .as_object_mut()
            .ok_or("invalid round plan structure")?
            .remove("content_hash");
        crate::storage::digest(&value)
    }
    fn persist(&mut self, path: &Path) -> Result<(), String> {
        self.content_hash = self.hash()?;
        let bytes = crate::storage::stable_bytes(self)?;
        if bytes.len() > crate::storage::MAX_METADATA_BYTES {
            return Err("round plan exceeds portable metadata limit".into());
        }
        crate::storage::save_atomic(path, &bytes)
    }
}

/// Stream family a [`Trial::create_with_setup`] closure may draw from.
pub const SETUP_STREAM_FAMILY: &str = "setup";

/// Run `setup` on a fresh trial stream set and check it advanced only the
/// setup family.
fn run_setup<W, F>(trial: &str, seed: u64, setup: F) -> Result<(W, Streams), String>
where
    F: FnOnce(&mut Streams) -> Result<W, String>,
{
    let origin = Streams::new(trial.to_owned(), seed);
    let mut streams = origin.clone();
    let world = setup(&mut streams)?;
    streams.checked_scope_progress(&origin, SETUP_STREAM_FAMILY)?;
    Ok((world, streams))
}

/// Re-run the setup closure of a trial made by [`Trial::create_with_setup`]
/// from its recorded trial ID and seed, and check that it reproduces the
/// genesis world and RNG state exactly.
///
/// This checks only the setup recorded in `manifest.json`: it does **not**
/// authenticate the event journal, snapshots or fork ancestry. Also run
/// `Store::inspect_read_only` (and `research::verify_fork` for forks) before
/// treating a trial as verified evidence.
pub fn verify_setup<W, F>(dir: impl AsRef<Path>, setup: F) -> Result<(), String>
where
    W: World + Serialize + DeserializeOwned,
    F: FnOnce(&mut Streams) -> Result<W, String>,
{
    let genesis: Genesis<W> = serde_json::from_slice(&crate::storage::read_metadata_limited(
        &dir.as_ref().join("manifest.json"),
    )?)
    .map_err(|e| e.to_string())?;
    if genesis.initial_streams.seed_domain.is_some() {
        return Err("setup is verified on the trial that ran it, not on a fork".into());
    }
    let (world, streams) = run_setup(&genesis.trial_id, genesis.initial_streams.root_seed, setup)?;
    if streams != genesis.initial_streams
        || crate::storage::digest(&world)? != crate::storage::digest(&genesis.initial_world)?
    {
        return Err("setup does not reproduce the recorded genesis world and RNG state".into());
    }
    Ok(())
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
        Self::create_from_genesis(dir, genesis, policy, backend)
    }
    /// Create a trial whose initial world is drawn from the trial's own named
    /// RNG streams, so randomized setup (initial labels, graphs, ...) is part of
    /// the audited genesis and reproducible from `name` and `seed` alone.
    ///
    /// `setup` may draw only from the [`SETUP_STREAM_FAMILY`] streams
    /// (`setup` or `setup/<name>`). The scheduler, agent and world streams
    /// stay at their origin, so setup never shifts a trial's later draws. The
    /// advanced setup positions are written to genesis, carried into forks,
    /// and can be re-derived with [`verify_setup`].
    #[allow(clippy::too_many_arguments)]
    pub fn create_with_setup<F>(
        dir: impl AsRef<Path>,
        name: &str,
        seed: u64,
        kind: &str,
        setup: F,
        policy: P,
        backend: Arc<dyn Backend>,
        agents: Vec<AgentId>,
        mode: Scheduling,
    ) -> Result<Self, String>
    where
        F: FnOnce(&mut Streams) -> Result<W, String>,
    {
        // Cheap preflight before a possibly expensive setup; `Store::create`
        // remains the authoritative, race-safe check.
        if dir.as_ref().exists() {
            return Err("trial path already exists".into());
        }
        let schedule = Schedule::new(mode, agents)?;
        let trial = TrialId::new(name, seed).0;
        let (world, streams) = run_setup(&trial, seed, setup)?;
        let genesis = Genesis {
            schema: SCHEMA,
            trial_id: trial,
            initial_world: world,
            initial_schedule: schedule,
            initial_streams: streams,
            world_kind: kind.into(),
            build: Some(BuildProvenance::current()),
        };
        Self::create_from_genesis(dir, genesis, policy, backend)
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
        Self::resume_mode(dir, kind, policy, backend, replay_only, false)
    }

    /// Bypasses historical prefix replay by trusting the latest snapshot.
    /// Run `Store::inspect_read_only` for full independent integrity checks.
    pub fn resume_trusted(
        dir: impl AsRef<Path>,
        kind: &str,
        policy: P,
        backend: Arc<dyn Backend>,
        replay_only: bool,
    ) -> Result<Self, String> {
        Self::resume_mode(dir, kind, policy, backend, replay_only, true)
    }

    fn resume_mode(
        dir: impl AsRef<Path>,
        kind: &str,
        policy: P,
        backend: Arc<dyn Backend>,
        replay_only: bool,
        trust_snapshot: bool,
    ) -> Result<Self, String> {
        let store = if trust_snapshot {
            Store::open_trusted(&dir)?
        } else {
            Store::open(&dir)?
        };
        if store.genesis.world_kind != kind {
            return Err("wrong registered world kind".into());
        }
        let mut effects = EffectJournal::open_existing(&dir)?;
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
            if let Err(e) = observer.committed_with_log_offset(entry, self.store.log_bytes()) {
                self.diagnostics.push(format!("observer failed: {e}"));
            }
        }
    }
    fn pending_round_path(&self) -> std::path::PathBuf {
        self.store.dir.join(ROUND_PLAN_FILE)
    }
    fn ensure_no_pending_round(&self) -> Result<(), String> {
        if self.pending_round_path().exists() {
            return Err("unfinished synchronous round: call run_round to complete it first".into());
        }
        Ok(())
    }
    /// Verify that all already-committed round actors agree with the frozen
    /// plan, preserving the same effect-call keys on recovery.
    fn validate_round_prefix(&self, plan: &RoundPlan) -> Result<usize, String> {
        if plan.schema != ROUND_PLAN_SCHEMA
            || plan.trial_id != self.store.genesis.trial_id
            || plan.hash()? != plan.content_hash
            || plan.actors.is_empty()
            || plan.actors.len() > 10_000
            || self.store.sequence < plan.start_sequence
            || self.store.sequence - plan.start_sequence > plan.actors.len() as u64
        {
            return Err("invalid or stale durable round plan".into());
        }
        for (i, actor) in plan.actors.iter().enumerate() {
            for (call, (key, spec)) in actor.specs.iter().enumerate() {
                let expected = spec.key(
                    &plan.trial_id,
                    plan.start_sequence
                        .checked_add(i as u64)
                        .ok_or("round sequence overflow")?,
                    actor.actor,
                    call as u32,
                )?;
                if expected != *key {
                    return Err("round plan effect identity changed".into());
                }
            }
        }
        let completed = (self.store.sequence - plan.start_sequence) as usize;
        if completed == 0 {
            if self.store.chain_head() != plan.start_chain_head {
                return Err("round starting chain head changed".into());
            }
            return Ok(0);
        }
        let mut expected = plan.start_chain_head.clone();
        let mut seen = 0;
        self.store.visit_events(|entry| {
            if entry.sequence >= plan.start_sequence {
                if seen >= completed
                    || entry.actor != plan.actors[seen].actor
                    || entry.previous_hash != expected
                {
                    return Err("interrupted round history does not match saved plan".into());
                }
                expected = entry.hash;
                seen += 1;
            }
            Ok(())
        })?;
        if seen != completed || expected != self.store.chain_head() {
            return Err("interrupted round chain head mismatch".into());
        }
        Ok(completed)
    }
    /// Scheduler control is an audited event, never a raw in-memory mutation.
    pub fn pause_actor(&mut self, actor: AgentId) -> Result<(), String> {
        self.ensure_no_pending_round()?;
        let entry = self.store.set_paused_event(actor, true)?;
        self.emit(&entry);
        Ok(())
    }
    pub fn resume_actor(&mut self, actor: AgentId) -> Result<(), String> {
        self.ensure_no_pending_round()?;
        let entry = self.store.set_paused_event(actor, false)?;
        self.emit(&entry);
        Ok(())
    }
    pub async fn step(&mut self) -> Result<Step<W>, String> {
        self.ensure_no_pending_round()?;
        let mut sched = self.store.schedule.clone();
        let mut rng = self.store.streams.clone();
        let Some(actor) = sched.select(&mut rng)? else {
            return Ok(Step::NoEligible);
        };
        // The attempt sequence, not only the accepted-world step, forms the
        // effect call key; failed attempts cannot reuse paid calls accidentally.
        let policy_streams = rng.clone();
        let result = match self.store.world.observe(actor) {
            Ok(obs) => match self.policy.requests_with_streams(actor, &obs, &mut rng) {
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
                            self.policy
                                .decide_with_streams(actor, &obs, &receipts, &mut rng)
                        }
                    }
                }
                Err(e) => Err(e),
            },
            Err(e) => Err(e),
        };
        rng.checked_scope_progress(&policy_streams, &format!("agent/{}", actor.0))?;
        match result {
            Ok(action) => {
                // Staged in place on the store's world; every path that does
                // not commit the event undoes it before going on.
                let before_apply = rng.clone();
                let applied = self
                    .store
                    .world
                    .apply_undoable(actor, action.clone(), &mut rng);
                if let Err(error) = rng.checked_scope_progress(&before_apply, "world") {
                    if let Ok(undo) = applied {
                        undo.undo(&mut self.store.world);
                    }
                    return Err(error);
                }
                let undo = match applied {
                    Ok(undo) => undo,
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
                        return Ok(Step::Rejected { actor, reason });
                    }
                };
                let append = self.store.commit_in_place(
                    actor,
                    action,
                    before_apply,
                    sched.clone(),
                    rng.clone(),
                );
                if append.is_err() {
                    undo.undo(&mut self.store.world);
                }
                match append {
                    Ok(entry) => {
                        self.emit(&entry);
                        Ok(Step::Accepted(Box::new(entry)))
                    }
                    Err(AppendError::TooLarge) => {
                        // Admission failed before any append. Preserve effects
                        // but record a rejected attempt so replay can progress.
                        let reason = crate::storage::EVENT_TOO_LARGE.to_string();
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
                    Err(error) => Err(error.to_string()),
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
    /// Two-phase synchronous round: each eligible agent observes the SAME
    /// pre-round world. Effects are resolved concurrently in bounded batches;
    /// actions then commit in deterministic scheduler order. Reducer writes
    /// are sequential and conflicts use that declared order.
    ///
    /// The round API currently supports the non-stream-aware `requests` and
    /// `decide` policy methods. For stochastic decisions use sequential step(),
    /// which records named RNG progression as part of each committed event.
    /// Persist a frozen round plan *before* any possibly billable effect.
    /// If interrupted, subsequent run_round (including after Trial::resume)
    /// completes the saved round instead of calculating new observations.
    /// Requires serializable observations exclusively for this API.
    pub async fn run_round(&mut self, max_actors: usize) -> Result<Vec<Step<W>>, String>
    where
        W::Observation: Serialize + DeserializeOwned,
    {
        let path = self.pending_round_path();
        let plan: RoundPlan = if path.exists() {
            serde_json::from_slice(&crate::storage::read_metadata_limited(&path)?)
                .map_err(|e| format!("cannot restore pending round: {e}"))?
        } else {
            let available = self
                .store
                .schedule
                .agents
                .len()
                .saturating_sub(self.store.schedule.paused.len());
            if max_actors == 0 || available == 0 {
                return Ok(Vec::new());
            }
            if max_actors > 10_000 {
                return Err("round exceeds 10,000 actors".into());
            }
            let limit = max_actors.min(available);
            let mut schedule = self.store.schedule.clone();
            let mut streams = self.store.streams.clone();
            let mut actors = Vec::with_capacity(limit);
            for i in 0..limit {
                let Some(actor) = schedule.select(&mut streams)? else {
                    break;
                };
                let sequence = self
                    .store
                    .sequence
                    .checked_add(i as u64)
                    .ok_or("round sequence overflow")?;
                let mut planned = PlannedActor {
                    actor,
                    observation: None,
                    reason: None,
                    specs: Vec::new(),
                };
                match self.store.world.observe(actor) {
                    Err(reason) => planned.reason = Some(reason),
                    Ok(obs) => {
                        planned.observation =
                            Some(serde_json::to_value(&obs).map_err(|e| e.to_string())?);
                        match self.policy.requests(actor, &obs) {
                            Err(reason) => planned.reason = Some(reason),
                            Ok(specs) => {
                                if specs.len() > 32
                                    || specs
                                        .iter()
                                        .filter(|s| matches!(s, EffectSpec::Tool { .. }))
                                        .count()
                                        > 8
                                {
                                    planned.reason =
                                        Some("round policy exceeds effect limits".into());
                                } else {
                                    for (index, spec) in specs.into_iter().enumerate() {
                                        let key = spec.key(
                                            &self.store.genesis.trial_id,
                                            sequence,
                                            actor,
                                            index as u32,
                                        )?;
                                        planned.specs.push((key, spec));
                                    }
                                }
                            }
                        }
                    }
                }
                actors.push(planned);
            }
            if actors.is_empty() {
                return Ok(Vec::new());
            }
            let mut fresh = RoundPlan {
                schema: ROUND_PLAN_SCHEMA,
                trial_id: self.store.genesis.trial_id.clone(),
                start_sequence: self.store.sequence,
                start_chain_head: self.store.chain_head().into(),
                actors,
                content_hash: String::new(),
            };
            fresh.persist(&path)?;
            fresh
        };
        let completed = self.validate_round_prefix(&plan)?;
        if completed == plan.actors.len() {
            // A checkpoint-durability round cannot drop its durable plan
            // before the accepted/rejected event bytes themselves are synced.
            self.store.sync_log()?;
            std::fs::remove_file(&path).map_err(|e| e.to_string())?;
            crate::storage::sync_trial_directory(&self.store.dir)?;
            return Ok(Vec::new());
        }

        // Dispatched effects for *every actor* have stable keys because both
        // the observations and specs come from a durable pre-dispatch plan.
        let mut all_specs = Vec::new();
        for planned in plan.actors.iter().skip(completed) {
            all_specs.extend(planned.specs.clone());
        }
        let mut receipts = std::collections::BTreeMap::new();
        let mut cursor = 0usize;
        while cursor < all_specs.len() {
            let mut end = cursor;
            let mut tools = 0;
            while end < all_specs.len() && end - cursor < 32 {
                let is_tool = matches!(&all_specs[end].1, EffectSpec::Tool { .. });
                if is_tool && tools == 8 {
                    break;
                }
                tools += usize::from(is_tool);
                end += 1;
            }
            let resolved = self
                .effects
                .resolve_all(
                    all_specs[cursor..end].to_vec(),
                    self.backend.clone(),
                    self.concurrency,
                    self.timeout,
                    self.replay_only,
                )
                .await?;
            for receipt in resolved {
                receipts.insert(receipt.key.clone(), receipt);
            }
            cursor = end;
        }

        let mut events = Vec::with_capacity(plan.actors.len() - completed);
        for planned in plan.actors.iter().skip(completed) {
            let actor = planned.actor;
            let mut scheduled = self.store.schedule.clone();
            let mut rng = self.store.streams.clone();
            if scheduled.select(&mut rng)? != Some(actor) {
                return Err("synchronous round scheduler drift".into());
            }
            let action = if let Some(reason) = &planned.reason {
                Err(reason.clone())
            } else {
                let obs: W::Observation = serde_json::from_value(
                    planned
                        .observation
                        .clone()
                        .ok_or("round observation missing")?,
                )
                .map_err(|e| format!("round observation format changed: {e}"))?;
                let mut ordered = Vec::new();
                for (key, _) in &planned.specs {
                    ordered.push(
                        receipts
                            .get(key)
                            .ok_or("round effect receipt missing")?
                            .clone(),
                    );
                }
                if ordered.iter().any(|r| {
                    matches!(
                        &r.result,
                        crate::effects::EffectResult::Failed { .. }
                            | crate::effects::EffectResult::UnsupportedCapability { .. }
                    )
                }) {
                    Err("round effect failed or unsupported".into())
                } else {
                    self.policy.decide(actor, &obs, &ordered)
                }
            };
            let event = match action {
                Ok(action) => {
                    // Staged in place, as in `step`.
                    let before_apply = rng.clone();
                    let apply = self
                        .store
                        .world
                        .apply_undoable(actor, action.clone(), &mut rng);
                    if let Err(error) = rng.checked_scope_progress(&before_apply, "world") {
                        if let Ok(undo) = apply {
                            undo.undo(&mut self.store.world);
                        }
                        return Err(error);
                    }
                    match apply {
                        Err(reason) => {
                            let entry = self.store.append(
                                actor,
                                Outcome::Rejected {
                                    reason: reason.clone(),
                                },
                                scheduled,
                                rng,
                            )?;
                            self.emit(&entry);
                            Step::Rejected { actor, reason }
                        }
                        Ok(undo) => {
                            let accepted = self.store.commit_in_place(
                                actor,
                                action,
                                before_apply,
                                scheduled.clone(),
                                rng.clone(),
                            );
                            if accepted.is_err() {
                                undo.undo(&mut self.store.world);
                            }
                            match accepted {
                                Ok(entry) => {
                                    self.emit(&entry);
                                    Step::Accepted(Box::new(entry))
                                }
                                Err(AppendError::TooLarge) => {
                                    let reason = crate::storage::EVENT_TOO_LARGE.to_string();
                                    let entry = self.store.append(
                                        actor,
                                        Outcome::Rejected {
                                            reason: reason.clone(),
                                        },
                                        scheduled,
                                        rng,
                                    )?;
                                    self.emit(&entry);
                                    Step::Rejected { actor, reason }
                                }
                                Err(error) => return Err(error.to_string()),
                            }
                        }
                    }
                }
                Err(reason) => {
                    let entry = self.store.append(
                        actor,
                        Outcome::Rejected {
                            reason: reason.clone(),
                        },
                        scheduled,
                        rng,
                    )?;
                    self.emit(&entry);
                    Step::Rejected { actor, reason }
                }
            };
            events.push(event);
        }
        // If cleanup fails, the next run validates the complete prefix and
        // removes the plan without launching any backend calls.
        self.store.sync_log()?;
        std::fs::remove_file(&path).map_err(|e| e.to_string())?;
        crate::storage::sync_trial_directory(&self.store.dir)?;
        Ok(events)
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

#[cfg(test)]
mod r6_interrupted_round_tests {
    use super::*;
    use crate::{
        runtime::{ScriptedBackend, sample_request},
        storage::Store,
    };
    use serde::{Deserialize, Serialize};
    use std::{
        fs,
        sync::{
            Arc,
            atomic::{AtomicU64, Ordering},
        },
    };

    static NEXT: AtomicU64 = AtomicU64::new(0);
    fn path() -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "hiyal-r6-interrupted-round-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
    }
    #[derive(Clone, Serialize, Deserialize)]
    struct Shared {
        notes: u64,
    }
    impl World for Shared {
        type Action = u64;
        type Observation = u64;
        fn observe(&self, _actor: AgentId) -> Result<u64, String> {
            Ok(self.notes)
        }
        fn apply(&mut self, _actor: AgentId, action: u64) -> Result<(), String> {
            self.notes += action;
            Ok(())
        }
    }
    struct FrozenPolicy;
    impl EffectPolicy<Shared> for FrozenPolicy {
        fn requests(&self, a: AgentId, obs: &u64) -> Result<Vec<EffectSpec>, String> {
            Ok(vec![sample_request(&format!(
                "agent {} sees {obs} notes",
                a.0
            ))])
        }
        fn decide(&self, _: AgentId, _: &u64, _: &[Receipt]) -> Result<u64, String> {
            Ok(1)
        }
    }
    struct CrashAfterFirst;
    impl Observer<Shared> for CrashAfterFirst {
        fn committed(&self, _: &Entry<Shared, u64>) -> Result<(), String> {
            panic!("simulate process interruption after durable first event")
        }
    }

    #[derive(Clone, Serialize, Deserialize)]
    struct Words {
        total: u64,
    }
    impl World for Words {
        type Action = String;
        type Observation = u64;
        fn observe(&self, _: AgentId) -> Result<u64, String> {
            Ok(self.total)
        }
        fn apply(&mut self, _: AgentId, action: String) -> Result<(), String> {
            self.total += action.len() as u64;
            Ok(())
        }
    }
    struct OversizedRoundAction;
    impl EffectPolicy<Words> for OversizedRoundAction {
        fn decide(&self, a: AgentId, _: &u64, _: &[Receipt]) -> Result<String, String> {
            if a.0 == 0 {
                Ok("X".repeat(crate::storage::MAX_EVENT_RECORD_BYTES))
            } else {
                Ok("ok".into())
            }
        }
    }
    #[tokio::test]
    async fn oversized_round_event_is_recorded_rejection_not_orphaned_receipt() {
        let dir = path();
        let mut trial = Trial::create(
            &dir,
            "oversized-round",
            91,
            "words.v1",
            Words { total: 0 },
            OversizedRoundAction,
            Arc::new(ScriptedBackend),
            vec![AgentId(0), AgentId(1)],
            Scheduling::RoundRobin,
        )
        .unwrap();
        let events = trial.run_round(2).await.unwrap();
        assert_eq!(events.len(), 2);
        assert!(matches!(events[0], Step::Rejected { .. }));
        assert!(matches!(events[1], Step::Accepted(_)));
        assert_eq!(trial.store.sequence(), 2);
        assert_eq!(trial.store.world().total, 2);
        assert!(!dir.join(ROUND_PLAN_FILE).exists());
        drop(trial);
        fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn checkpoint_durability_round_syncs_before_deleting_plan() {
        let dir = path();
        let mut trial = Trial::create(
            &dir,
            "durable-round",
            87,
            "shared.v1",
            Shared { notes: 0 },
            FrozenPolicy,
            Arc::new(ScriptedBackend),
            vec![AgentId(0), AgentId(1)],
            Scheduling::RoundRobin,
        )
        .unwrap();
        trial
            .store
            .set_durability(crate::storage::Durability::AtCheckpoint)
            .unwrap();
        let events = trial.run_round(2).await.unwrap();
        assert_eq!(events.len(), 2);
        assert!(!dir.join(ROUND_PLAN_FILE).exists());
        // Deliberately NO checkpoint. The round itself must have fsynced
        // all authoritative events before publishing plan removal.
        drop(trial);
        let resumed = Trial::resume(
            &dir,
            "shared.v1",
            FrozenPolicy,
            Arc::new(ScriptedBackend),
            false,
        )
        .unwrap();
        assert_eq!(resumed.store.sequence(), 2);
        assert_eq!(resumed.store.world().notes, 2);
        assert_eq!(resumed.effects.receipts().len(), 2);
        drop(resumed);
        fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn crash_after_paid_round_receipts_never_reissues_call() {
        let dir = path();
        let mut trial = Trial::create(
            &dir,
            "crashed-round",
            71,
            "shared.v1",
            Shared { notes: 0 },
            FrozenPolicy,
            Arc::new(ScriptedBackend),
            vec![AgentId(0), AgentId(1)],
            Scheduling::RoundRobin,
        )
        .unwrap();
        trial.observe_with(Arc::new(CrashAfterFirst));
        let task = tokio::spawn(async move { trial.run_round(2).await });
        assert!(task.await.err().is_some_and(|error| error.is_panic()));
        assert!(dir.join(ROUND_PLAN_FILE).exists());
        let mut resumed = Trial::resume(
            &dir,
            "shared.v1",
            FrozenPolicy,
            Arc::new(ScriptedBackend),
            false,
        )
        .unwrap();
        assert_eq!(resumed.store.sequence(), 1);
        assert_eq!(resumed.effects.receipts().len(), 2);
        assert!(
            resumed.step().await.is_err(),
            "cannot re-plan before completing frozen round"
        );
        let remainder = resumed.run_round(2).await.unwrap();
        assert_eq!(remainder.len(), 1);
        assert_eq!(
            resumed.effects.receipts().len(),
            2,
            "paid call was re-dispatched"
        );
        assert_eq!(resumed.store.world().notes, 2);
        assert!(!dir.join(ROUND_PLAN_FILE).exists());
        resumed.checkpoint().unwrap();
        drop(resumed);
        let verified = Store::<Shared, u64>::inspect_read_only(&dir).unwrap();
        assert_eq!(verified.sequence, 2);
        fs::remove_dir_all(dir).unwrap();
    }
}
