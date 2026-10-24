//! Reproducible research batches and paired committed-boundary forks.
//! No Python runtime, global random generators, or implicit network work.

use crate::{
    builtins::{Counter, Increment, kind},
    core::{AgentId, TrialId, World},
    experiment::{EffectPolicy, Trial},
    metrics::{CsvMetrics, Metric, MetricProjector},
    runtime::{Backend, ScriptedBackend},
    scheduler::Scheduling,
    storage::{Genesis, ReadOnly, SCHEMA, Store, digest, stable_bytes},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    fs::{self, File},
    future::Future,
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::task::JoinSet;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sweep {
    pub name: String,
    pub base_seed: u64,
    pub parameter_values: Vec<u64>,
    pub replicates: u32,
    pub actors: u32,
    pub attempts: u64,
    pub max_parallel: usize,
    pub scheduler: Scheduling,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Job {
    pub id: String,
    pub trial_name: String,
    pub seed: u64,
    pub parameter: u64,
    pub replicate: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobResult {
    pub job: Job,
    pub trial_id: String,
    pub sequence: u64,
    pub final_value: u64,
    pub chain_head: String,
}

/// Resolve existing symlinks in the longest available ancestor, then
/// append any not-yet-created path components. This checks containment
/// without first materializing an unsafe child under the source.
fn canonical_future_path(path: &Path) -> Result<PathBuf, String> {
    // Anchor relative paths at the actual working directory so a
    // one-component destination such as "branch" can walk to an existing
    // parent instead of reaching Path::parent() == "".
    let mut ancestor = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| e.to_string())?
            .join(path)
    };
    let mut missing = Vec::new();
    while !ancestor.exists() {
        let name = ancestor
            .file_name()
            .ok_or("destination contains unresolved or invalid path components")?;
        missing.push(name.to_os_string());
        ancestor = ancestor
            .parent()
            .ok_or("destination has no existing ancestor")?
            .to_path_buf();
    }
    let mut resolved = fs::canonicalize(&ancestor).map_err(|e| e.to_string())?;
    for name in missing.iter().rev() {
        resolved.push(name);
    }
    Ok(resolved)
}

fn simple_name(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 100
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

impl Sweep {
    pub fn jobs(&self) -> Result<Vec<Job>, String> {
        if !simple_name(&self.name)
            || self.parameter_values.is_empty()
            || self.parameter_values.len() > 64
            || self.replicates == 0
            || self.replicates > 64
            || self.actors == 0
            || self.actors > 1000
            || self.attempts > 100_000
            || self.max_parallel == 0
            || self.max_parallel > 16
            || self.parameter_values.len() * self.replicates as usize > 512
        {
            return Err("invalid sweep name or resource limits".into());
        }
        let mut seen = std::collections::BTreeSet::new();
        if !self.parameter_values.iter().all(|v| seen.insert(v)) {
            return Err("duplicate sweep parameter".into());
        }
        // Stable research identity covers every scientific setting while
        // excluding the operational worker count (max_parallel).
        let identity = digest(&(
            "r2-sweep-identity-v2",
            &self.name,
            self.base_seed,
            &self.parameter_values,
            self.replicates,
            self.actors,
            self.attempts,
            self.scheduler,
        ))?;
        let mut jobs = Vec::new();
        for parameter in &self.parameter_values {
            for replicate in 0..self.replicates {
                let seed_hex =
                    crate::core::derive_id("r2-replicate", &[&identity, &replicate.to_string()]);
                let seed = u64::from_str_radix(&seed_hex[..16], 16).map_err(|e| e.to_string())?;
                let id = crate::core::derive_id(
                    "r2-job",
                    &[
                        &identity,
                        &parameter.to_string(),
                        &replicate.to_string(),
                        &seed.to_string(),
                    ],
                );
                jobs.push(Job {
                    trial_name: format!(
                        "{}-p{}-r{}-{}",
                        self.name,
                        parameter,
                        replicate,
                        &identity[..16]
                    ),
                    id,
                    seed,
                    parameter: *parameter,
                    replicate,
                });
            }
        }
        Ok(jobs)
    }

    /// Job function runs only inside bounded worker slots. A failure stops
    /// admitting new jobs but drains work already started.
    pub async fn execute<T, F, Fut>(&self, action: F) -> Result<Vec<(Job, T)>, String>
    where
        T: Send + 'static,
        F: Fn(Job) -> Fut + Clone + Send + Sync + 'static,
        Fut: Future<Output = Result<T, String>> + Send + 'static,
    {
        let jobs = self.jobs()?;
        let mut iterator = jobs.into_iter();
        let mut active = JoinSet::new();
        let mut finished = Vec::new();
        let mut errors = Vec::new();
        for _ in 0..self.max_parallel {
            if let Some(job) = iterator.next() {
                let fun = action.clone();
                active.spawn(async move {
                    let value = fun(job.clone()).await;
                    (job, value)
                });
            }
        }
        while let Some(result) = active.join_next().await {
            match result {
                Ok((job, Ok(value))) => {
                    finished.push((job, value));
                    if errors.is_empty()
                        && let Some(next) = iterator.next()
                    {
                        let fun = action.clone();
                        active.spawn(async move {
                            let value = fun(next.clone()).await;
                            (next, value)
                        });
                    }
                }
                Ok((job, Err(error))) => errors.push(format!("job {}: {error}", job.id)),
                Err(error) => errors.push(format!("research worker crashed: {error}")),
            }
        }
        if !errors.is_empty() {
            return Err(errors.join("; "));
        }
        finished.sort_by(|a, b| a.0.id.cmp(&b.0.id));
        Ok(finished)
    }

    /// Built-in end-to-end reference runner; idempotently resumes every trial
    /// to the requested total number of attempted steps. No repeated steps.
    pub async fn run_counter(&self, root: impl AsRef<Path>) -> Result<Vec<JobResult>, String> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root).map_err(|e| e.to_string())?;
        // Generic Sweep::execute accepts arbitrary scientific parameters;
        // only this fixed counter reference needs multiplication checked.
        let _ = self.jobs()?;
        if self
            .parameter_values
            .iter()
            .any(|amount| amount.checked_mul(self.attempts).is_none())
        {
            return Err("counter amount multiplied by attempts overflows u64".into());
        }
        let plan_path = root.join("sweep-plan.json");
        // Worker count is an operational resource limit, not scientific
        // experiment identity. Switching 1 -> 16 workers must be restartable.
        let planned = stable_bytes(&(
            "r2-plan-v1",
            &self.name,
            self.base_seed,
            &self.parameter_values,
            self.replicates,
            self.actors,
            self.attempts,
            self.scheduler,
        ))?;
        if plan_path.exists() {
            if fs::read(&plan_path).map_err(|e| e.to_string())? != planned {
                return Err("cannot restart a sweep with different parameters".into());
            }
        } else {
            crate::storage::save_atomic(&plan_path, &planned)?;
        }
        let settings = self.clone();
        let job_root = root.clone();
        let records = self
            .execute(move |job: Job| {
                let root = job_root.clone();
                let settings = settings.clone();
                async move {
                    let dir = root.join(&job.id);
                    let world_kind = kind(job.parameter, false, "fixed-v1", "scripted")?;
                    let policy = Increment {
                        amount: job.parameter,
                        model: false,
                        model_id: "fixed-v1".into(),
                        provider: "scripted".into(),
                    };
                    let mut trial = if dir.exists() {
                        let trial = Trial::resume(
                            &dir,
                            &world_kind,
                            policy,
                            Arc::new(ScriptedBackend),
                            false,
                        )?;
                        if trial.store.genesis().trial_id
                            != TrialId::new(&job.trial_name, job.seed).0
                            || trial.store.genesis().initial_schedule.mode != settings.scheduler
                            || trial.store.genesis().initial_schedule.agents.len()
                                != settings.actors as usize
                            || trial.store.sequence() > settings.attempts
                        {
                            return Err("resumed sweep configuration/progress mismatch".into());
                        }
                        trial
                    } else {
                        Trial::create(
                            &dir,
                            &job.trial_name,
                            job.seed,
                            &world_kind,
                            Counter { total: 0 },
                            policy,
                            Arc::new(ScriptedBackend),
                            (0..settings.actors as u64).map(AgentId).collect(),
                            settings.scheduler,
                        )?
                    };
                    let projector = Arc::new(CsvMetrics::recover(&dir, CounterMetrics)?);
                    trial.observe_with(projector);
                    trial
                        .run_attempts(settings.attempts - trial.store.sequence())
                        .await?;
                    ensure_observer_health(&trial.diagnostics)?;
                    trial.checkpoint()?;
                    Ok(JobResult {
                        trial_id: trial.store.genesis().trial_id.clone(),
                        sequence: trial.store.sequence(),
                        final_value: trial.store.world().total,
                        chain_head: trial.store.chain_head().into(),
                        job,
                    })
                }
            })
            .await?;
        let result: Vec<_> = records.into_iter().map(|(_, value)| value).collect();
        let mut csv =
            String::from("job_id,trial_id,parameter,replicate,seed,attempts,total,chain_head\n");
        for row in &result {
            csv.push_str(&format!(
                "{},{},{},{},{},{},{},{}\n",
                row.job.id,
                row.trial_id,
                row.job.parameter,
                row.job.replicate,
                row.job.seed,
                row.sequence,
                row.final_value,
                row.chain_head
            ));
        }
        let path = root.join("results.csv");
        crate::storage::save_atomic(&path, csv.as_bytes())?;
        Ok(result)
    }
}

/// Observers do not roll back authoritative committed events, but a research
/// runner must not publish a successful derived dataset after observer errors.
pub fn ensure_observer_health(diagnostics: &[String]) -> Result<(), String> {
    if diagnostics.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "derived observer output failed; restart can rebuild projections: {}",
            diagnostics.join("; ")
        ))
    }
}

pub struct CounterMetrics;
impl MetricProjector<Counter> for CounterMetrics {
    fn project(&self, world: &Counter) -> Result<Vec<Metric>, String> {
        Ok(vec![Metric {
            name: "total".into(),
            value: world.total.to_string(),
        }])
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ForkMetadata {
    pub version: u32,
    pub branch_id: String,
    pub parent_id: String,
    pub parent_head: String,
    pub parent_sequence: u64,
    pub parent_logical_step: u64,
    pub source_manifest_sha256: String,
    pub source_events_sha256: String,
    pub pre_state_hash: String,
    pub post_state_hash: String,
    pub intervention: String,
    pub arm: String,
    pub parameters: serde_json::Value,
    pub attribution_hash: String,
}
impl ForkMetadata {
    fn fingerprint(&self) -> Result<String, String> {
        let mut copy = self.clone();
        copy.attribution_hash.clear();
        digest(&copy)
    }
}

fn file_sha256(path: &Path) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    let mut sha = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let count = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        sha.update(&buffer[..count]);
    }
    Ok(crate::core::hex_digest(&sha.finalize()))
}

/// fs::copy keeps the source's permissions, so a copy of read-only evidence
/// is read-only too. A read handle is enough to fsync on Unix.
#[cfg(not(windows))]
fn sync_copied_file(path: &Path) -> Result<(), String> {
    File::open(path)
        .and_then(|f| f.sync_all())
        .map_err(|e| e.to_string())
}

/// FlushFileBuffers on Windows requires a writable handle, so the read-only
/// attribute copied from the source is cleared on our own copy first.
#[cfg(windows)]
fn sync_copied_file(path: &Path) -> Result<(), String> {
    let mut permissions = fs::metadata(path).map_err(|e| e.to_string())?.permissions();
    if permissions.readonly() {
        permissions.set_readonly(false);
        fs::set_permissions(path, permissions).map_err(|e| e.to_string())?;
    }
    fs::OpenOptions::new()
        .write(true)
        .open(path)
        .and_then(|f| f.sync_all())
        .map_err(|e| e.to_string())
}

fn copy_evidence(source: &Path, dest: &Path) -> Result<(), String> {
    fs::create_dir_all(dest).map_err(|e| e.to_string())?;
    for name in ["manifest.json", "events.jsonl"] {
        let to = dest.join(name);
        fs::copy(source.join(name), &to).map_err(|e| e.to_string())?;
        sync_copied_file(&to)?;
    }
    crate::storage::sync_trial_directory(dest)
}

/// Verify a self-contained ancestral log and the boundary attribution for
/// the child. Neither fork depends on the *live* parent after creation.
pub fn verify_fork<W>(child: impl AsRef<Path>) -> Result<ForkMetadata, String>
where
    W: World + Serialize + DeserializeOwned + 'static,
    W::Action: Serialize + DeserializeOwned + Clone + 'static,
{
    verify_fork_views::<W>(child).map(|fork| fork.metadata)
}

/// A fork that passed [`verify_fork`], with the two views it verified.
pub struct VerifiedFork<W> {
    pub metadata: ForkMetadata,
    /// Full read-only inspection of the fork's ancestry copy of its parent.
    pub parent: ReadOnly<W>,
    /// Full read-only inspection of the fork itself.
    pub branch: ReadOnly<W>,
}

/// [`verify_fork`], also returning the parent and branch views it fully
/// verified, so a caller that goes on to read the branch does not inspect
/// it again.
///
/// Like every `inspect_read_only_cached` call, this reuses the in-process
/// memo only for byte-identical evidence this process already fully
/// verified; any change to a file is replayed and checked again. A final
/// evidence gate that must be independent of earlier work in the same
/// process should run in a fresh process, or after
/// [`crate::storage::clear_replay_cache`].
pub fn verify_fork_views<W>(child: impl AsRef<Path>) -> Result<VerifiedFork<W>, String>
where
    W: World + Serialize + DeserializeOwned + 'static,
    W::Action: Serialize + DeserializeOwned + Clone + 'static,
{
    let child = child.as_ref();
    let raw = fs::read(child.join("fork.json")).map_err(|e| e.to_string())?;
    let meta: ForkMetadata = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
    if meta.version != 1 || meta.attribution_hash != meta.fingerprint()? {
        return Err("fork attribution checksum mismatch".into());
    }
    let bundle = child.join("ancestry");
    if file_sha256(&bundle.join("manifest.json"))? != meta.source_manifest_sha256
        || file_sha256(&bundle.join("events.jsonl"))? != meta.source_events_sha256
    {
        return Err("parent evidence bytes altered".into());
    }
    let parent: ReadOnly<W> = Store::<W, W::Action>::inspect_read_only_cached(&bundle)?;
    let branch: ReadOnly<W> = Store::<W, W::Action>::inspect_read_only_cached(child)?;
    if parent.genesis.trial_id != meta.parent_id
        || parent.chain_head != meta.parent_head
        || parent.sequence != meta.parent_sequence
        || parent.logical_step != meta.parent_logical_step
        || digest(&parent.world)? != meta.pre_state_hash
        || digest(&branch.genesis.initial_world)? != meta.post_state_hash
        || branch.genesis.trial_id != meta.branch_id
        || branch.genesis.initial_schedule != parent.schedule
        || branch.genesis.initial_streams.positions != parent.streams.positions
        || branch.genesis.initial_streams.root_seed != parent.streams.root_seed
        || branch.genesis.initial_streams.profile != parent.streams.profile
        || branch.genesis.initial_streams.seed_domain.as_deref()
            != Some(
                parent
                    .streams
                    .seed_domain
                    .as_deref()
                    .unwrap_or(&parent.streams.trial),
            )
        || branch.genesis.world_kind != parent.genesis.world_kind
    {
        return Err("fork boundary, schedule, RNG, or intervention attribution mismatch".into());
    }
    Ok(VerifiedFork {
        metadata: meta,
        parent,
        branch,
    })
}

/// Create a branch at the validated *latest committed boundary*. This
/// intentionally does not support a mid-event fork. Intervention applies once
/// to genesis, and its before/after state plus parameters are recorded.
#[allow(clippy::too_many_arguments)]
pub fn fork_at_head<W, P, F>(
    parent_dir: impl AsRef<Path>,
    child_dir: impl AsRef<Path>,
    intervention: &str,
    arm: &str,
    parameters: serde_json::Value,
    intervene: F,
    policy: P,
    backend: Arc<dyn Backend>,
) -> Result<Trial<W, P>, String>
where
    W: World + Serialize + DeserializeOwned + 'static,
    W::Action: Serialize + DeserializeOwned + Clone + 'static,
    P: EffectPolicy<W>,
    F: FnOnce(&mut W) -> Result<(), String>,
{
    if !simple_name(intervention) || !simple_name(arm) {
        return Err("invalid intervention or arm identifier".into());
    }
    let parent_dir = parent_dir.as_ref();
    let child_dir = child_dir.as_ref();
    let canonical_parent = fs::canonicalize(parent_dir).map_err(|e| e.to_string())?;
    let canonical_child = canonical_future_path(child_dir)?;
    if child_dir.exists()
        || canonical_child.starts_with(&canonical_parent)
        || canonical_parent.starts_with(&canonical_child)
    {
        return Err("branch directory must be new and independent of parent".into());
    }
    let parent: ReadOnly<W> = Store::<W, W::Action>::inspect_read_only_cached(parent_dir)?;
    let before = digest(&parent.world)?;
    let mut world = parent.world.clone();
    intervene(&mut world)?;
    let after = digest(&world)?;
    // Different parent boundaries, parameters, and post-intervention states
    // must never share trial IDs or model-effect call identities.
    let attribution = digest(&(
        "r2-fork-id-v2",
        &parent.genesis.trial_id,
        &parent.chain_head,
        parent.sequence,
        intervention,
        arm,
        &parameters,
        &before,
        &after,
    ))?;
    let name = format!("fork:{}:{attribution}", parent.genesis.trial_id);
    let branch_id = TrialId::new(&name, parent.streams.root_seed).0;
    let streams = parent.streams.paired_branch(branch_id.clone());
    let genesis = Genesis {
        schema: SCHEMA,
        trial_id: branch_id.clone(),
        initial_world: world,
        initial_schedule: parent.schedule.clone(),
        initial_streams: streams,
        world_kind: parent.genesis.world_kind.clone(),
        build: Some(crate::storage::BuildProvenance::current()),
    };
    // Build an entire branch in a unique sibling directory. The public child
    // path appears only after ancestry, attribution and both ledgers verify.
    // A process kill leaves an ignorable orphan staging directory, not a
    // partially published branch that blocks restart.
    static STAGE_SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let parent_folder = child_dir
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent_folder).map_err(|e| e.to_string())?;
    let basename = child_dir
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or("invalid fork destination name")?;
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let sequence = STAGE_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let stage = parent_folder.join(format!(
        ".{basename}.pending-{}-{nonce}-{sequence}",
        std::process::id()
    ));
    let kind = genesis.world_kind.clone();
    let mut stage_owned = false;
    let create = (|| -> Result<(), String> {
        let store = Store::<W, W::Action>::create(&stage, genesis)?;
        stage_owned = true;
        let mut ledger = crate::runtime::EffectJournal::open(&stage)?;
        ledger.bind_trial(&store.genesis().trial_id)?;
        drop(ledger);
        drop(store);
        let bundle = stage.join("ancestry");
        copy_evidence(parent_dir, &bundle).map_err(|e| format!("copy ancestral evidence: {e}"))?;
        // verify_fork below independently reconstructs ancestry once. An
        // exact log length check here rejects a changed parent prefix without
        // pointlessly replaying the captured log an extra time.
        let captured_length = fs::metadata(bundle.join("events.jsonl"))
            .map_err(|e| e.to_string())?
            .len();
        if captured_length != parent.log_bytes {
            return Err("parent advanced during branch evidence capture".into());
        }
        let mut meta = ForkMetadata {
            version: 1,
            branch_id,
            parent_id: parent.genesis.trial_id,
            parent_head: parent.chain_head,
            parent_sequence: parent.sequence,
            parent_logical_step: parent.logical_step,
            source_manifest_sha256: file_sha256(&bundle.join("manifest.json"))?,
            source_events_sha256: file_sha256(&bundle.join("events.jsonl"))?,
            pre_state_hash: before,
            post_state_hash: after,
            intervention: intervention.into(),
            arm: arm.into(),
            parameters,
            attribution_hash: String::new(),
        };
        meta.attribution_hash = meta.fingerprint()?;
        crate::storage::save_atomic(&stage.join("fork.json"), &stable_bytes(&meta)?)?;
        verify_fork::<W>(&stage).map_err(|e| format!("verify staged fork: {e}"))?;
        if child_dir.exists() {
            return Err("fork destination appeared during creation".into());
        }
        fs::rename(&stage, child_dir).map_err(|e| format!("publish staged fork: {e}"))?;
        crate::storage::sync_trial_directory(parent_folder)?;
        Ok(())
    })();
    if let Err(error) = create {
        if stage_owned {
            let _ = fs::remove_dir_all(&stage);
        }
        return Err(error);
    }
    Trial::resume(child_dir, &kind, policy, backend, false)
        .map_err(|e| format!("open published fork: {e}"))
}

/// Control and treatment trials returned by `paired_forks`.
pub type ForkPair<W, P> = (Trial<W, P>, Trial<W, P>);

/// Paired branches share an independently verified parent boundary and RNG
/// stream domain, but have different trial IDs, histories, and effect ledgers.
pub fn paired_forks<W, P, F>(
    parent: impl AsRef<Path>,
    root: impl AsRef<Path>,
    intervention: &str,
    parameters: serde_json::Value,
    make_policy: impl Fn() -> P,
    treatment: F,
    backend: Arc<dyn Backend>,
) -> Result<ForkPair<W, P>, String>
where
    W: World + Serialize + DeserializeOwned + 'static,
    W::Action: Serialize + DeserializeOwned + Clone + 'static,
    P: EffectPolicy<W>,
    F: FnOnce(&mut W) -> Result<(), String>,
{
    let root = root.as_ref();
    fs::create_dir_all(root).map_err(|e| e.to_string())?;
    let control_dir = root.join("control");
    let treatment_dir = root.join("treatment");
    let control = fork_at_head(
        parent.as_ref(),
        &control_dir,
        intervention,
        "control",
        serde_json::json!({}),
        |_world: &mut W| Ok(()),
        make_policy(),
        backend.clone(),
    )?;
    let treatment = match fork_at_head(
        parent.as_ref(),
        &treatment_dir,
        intervention,
        "treatment",
        parameters,
        treatment,
        make_policy(),
        backend,
    ) {
        Ok(value) => value,
        Err(error) => {
            drop(control);
            let _ = fs::remove_dir_all(&control_dir);
            return Err(error);
        }
    };
    // fork_at_head independently validated both arms before publication.
    // Compare compact committed attribution metadata rather than replaying
    // the complete copied prefix a third time for each new arm.
    let paired_check = (|| -> Result<(), String> {
        let left: ForkMetadata = serde_json::from_slice(
            &fs::read(control_dir.join("fork.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let right: ForkMetadata = serde_json::from_slice(
            &fs::read(treatment_dir.join("fork.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        if left.parent_id != right.parent_id
            || left.parent_head != right.parent_head
            || left.parent_sequence != right.parent_sequence
            || left.parent_logical_step != right.parent_logical_step
            || left.arm != "control"
            || right.arm != "treatment"
            || control.store.genesis().initial_streams.seed_domain
                != treatment.store.genesis().initial_streams.seed_domain
            || control.store.genesis().initial_schedule
                != treatment.store.genesis().initial_schedule
        {
            return Err("paired branches differ in ancestry, RNG or schedule".into());
        }
        Ok(())
    })();
    if let Err(error) = paired_check {
        drop(control);
        drop(treatment);
        let _ = fs::remove_dir_all(&control_dir);
        let _ = fs::remove_dir_all(&treatment_dir);
        return Err(error);
    }
    Ok((control, treatment))
}

/// Resume a fork only after verifying the copied ancestral history and
/// intervention metadata. Restart does not reapply the intervention.
pub fn resume_fork<W, P>(
    dir: impl AsRef<Path>,
    policy: P,
    backend: Arc<dyn Backend>,
    replay_only: bool,
) -> Result<Trial<W, P>, String>
where
    W: World + Serialize + DeserializeOwned + 'static,
    W::Action: Serialize + DeserializeOwned + Clone + 'static,
    P: EffectPolicy<W>,
{
    verify_fork::<W>(&dir)?;
    let genesis: Genesis<W> = serde_json::from_slice(&crate::storage::read_metadata_limited(
        &dir.as_ref().join("manifest.json"),
    )?)
    .map_err(|e| e.to_string())?;
    Trial::resume(dir, &genesis.world_kind, policy, backend, replay_only)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    fn directory(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("hiyal-r2-{label}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[cfg(unix)]
    #[test]
    fn copy_evidence_accepts_read_only_parent_files() {
        use std::os::unix::fs::PermissionsExt;
        let source = directory("read-only-parent");
        let dest = directory("read-only-copy");
        fs::create_dir_all(&source).unwrap();
        for name in ["manifest.json", "events.jsonl"] {
            let path = source.join(name);
            fs::write(&path, b"{}\n").unwrap();
            fs::set_permissions(&path, fs::Permissions::from_mode(0o444)).unwrap();
        }
        copy_evidence(&source, &dest).unwrap();
        assert_eq!(fs::read(dest.join("events.jsonl")).unwrap(), b"{}\n");
        fs::remove_dir_all(source).unwrap();
        fs::remove_dir_all(dest).unwrap();
    }

    fn sweep(parallel: usize) -> Sweep {
        Sweep {
            name: "survey".into(),
            base_seed: 122,
            parameter_values: vec![1, 2],
            replicates: 3,
            actors: 3,
            attempts: 9,
            max_parallel: parallel,
            scheduler: Scheduling::Random,
        }
    }
    #[test]
    fn scientific_sweep_identity_changes_without_worker_count_aliasing() {
        let baseline = sweep(1);
        let base_job = baseline.jobs().unwrap().remove(0);
        let mut workers = baseline.clone();
        workers.max_parallel = 4;
        assert_eq!(base_job, workers.jobs().unwrap().remove(0));
        for changed in 0..4 {
            let mut study = baseline.clone();
            match changed {
                0 => study.actors += 1,
                1 => study.attempts += 1,
                2 => study.scheduler = Scheduling::RoundRobin,
                _ => study.parameter_values.push(3),
            }
            let other = study.jobs().unwrap().remove(0);
            assert_ne!(base_job.id, other.id);
            assert_ne!(
                TrialId::new(&base_job.trial_name, base_job.seed),
                TrialId::new(&other.trial_name, other.seed)
            );
        }
    }

    #[tokio::test]
    async fn sweep_replays_identically_across_parallelism_and_restarts() {
        let a = directory("serial");
        let b = directory("parallel");
        let one = sweep(1).run_counter(&a).await.unwrap();
        let many = sweep(3).run_counter(&b).await.unwrap();
        assert_eq!(one, many);
        assert_eq!(one.len(), 6);
        let again = sweep(2).run_counter(&b).await.unwrap();
        assert_eq!(many, again);
        for job in &again {
            assert_eq!(job.sequence, 9);
            assert_eq!(job.final_value, job.job.parameter * 9);
            assert_eq!(
                fs::read(a.join(&job.job.id).join("metrics.csv")).unwrap(),
                fs::read(b.join(&job.job.id).join("metrics.csv")).unwrap()
            );
        }
        fs::remove_dir_all(a).unwrap();
        fs::remove_dir_all(b).unwrap();
    }

    #[tokio::test]
    async fn worker_pool_enforces_cap_and_cancels_no_inflight_tasks_on_error() {
        let mut plan = sweep(2);
        plan.parameter_values = vec![1, 2, 3];
        plan.replicates = 2;
        let current = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let count = Arc::new(AtomicUsize::new(0));
        let run = plan
            .execute({
                let current = current.clone();
                let peak = peak.clone();
                let count = count.clone();
                move |job| {
                    let current = current.clone();
                    let peak = peak.clone();
                    let count = count.clone();
                    async move {
                        let running = current.fetch_add(1, Ordering::SeqCst) + 1;
                        peak.fetch_max(running, Ordering::SeqCst);
                        count.fetch_add(1, Ordering::SeqCst);
                        tokio::time::sleep(Duration::from_millis(3)).await;
                        current.fetch_sub(1, Ordering::SeqCst);
                        if job.parameter == 1 {
                            Err("synthetic failure".into())
                        } else {
                            Ok(job.parameter)
                        }
                    }
                }
            })
            .await;
        assert!(run.is_err());
        assert!(peak.load(Ordering::SeqCst) <= 2);
        assert_eq!(current.load(Ordering::SeqCst), 0);
        assert!(count.load(Ordering::SeqCst) < 6);
    }

    #[tokio::test]
    async fn oversized_counter_parameters_are_rejected_only_by_counter_runner() {
        let mut plan = sweep(1);
        plan.parameter_values = vec![u64::MAX];
        plan.attempts = 2;
        assert!(plan.jobs().is_ok());
        let custom = plan
            .execute(|job| async move { Ok(job.parameter) })
            .await
            .unwrap();
        assert_eq!(custom.len(), 3);
        let path = directory("counter-overflow");
        assert!(plan.run_counter(&path).await.is_err());
        assert!(!path.join("sweep-plan.json").exists());
        fs::remove_dir_all(path).unwrap();
        plan.attempts = 1;
        assert!(plan.jobs().is_ok());
    }

    #[test]
    fn incomplete_derived_metrics_cannot_be_reported_as_success() {
        assert!(ensure_observer_health(&[]).is_ok());
        assert!(ensure_observer_health(&["observer failed: disk full".into()]).is_err());
    }

    #[tokio::test]
    async fn failed_second_arm_rolls_back_published_first_arm() {
        use std::sync::atomic::AtomicUsize;
        let root = directory("second-arm-failure");
        let parent = root.join("parent");
        let kind = kind(1, false, "fixed-v1", "scripted").unwrap();
        {
            let trial = Trial::create(
                &parent,
                "parent-for-rollback",
                5,
                &kind,
                Counter { total: 0 },
                Increment {
                    amount: 1,
                    model: false,
                    model_id: "fixed-v1".into(),
                    provider: "scripted".into(),
                },
                Arc::new(ScriptedBackend),
                vec![AgentId(0)],
                Scheduling::RoundRobin,
            )
            .unwrap();
            drop(trial);
        }
        let changed = Arc::new(AtomicUsize::new(0));
        let parent_clone = parent.clone();
        let kind_clone = kind.clone();
        let result = paired_forks::<Counter, _, _>(
            &parent,
            root.join("pair"),
            "test",
            serde_json::json!({"delta":1}),
            {
                let changed = changed.clone();
                move || {
                    if changed.fetch_add(1, Ordering::SeqCst) == 1 {
                        let mut source = Trial::resume(
                            &parent_clone,
                            &kind_clone,
                            Increment {
                                amount: 1,
                                model: false,
                                model_id: "fixed-v1".into(),
                                provider: "scripted".into(),
                            },
                            Arc::new(ScriptedBackend),
                            false,
                        )
                        .unwrap();
                        source.pause_actor(AgentId(0)).unwrap();
                        source.checkpoint().unwrap();
                    }
                    Increment {
                        amount: 1,
                        model: false,
                        model_id: "fixed-v1".into(),
                        provider: "scripted".into(),
                    }
                }
            },
            |world: &mut Counter| {
                world.total += 1;
                Ok(())
            },
            Arc::new(ScriptedBackend),
        );
        assert!(result.is_err());
        assert!(!root.join("pair/control").exists());
        assert!(!root.join("pair/treatment").exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn new_single_component_relative_fork_destination_is_supported() {
        let name = format!("hiyal-r2-relative-{}", std::process::id());
        let path = canonical_future_path(Path::new(&name)).unwrap();
        let cwd = std::env::current_dir().unwrap().canonicalize().unwrap();
        assert_eq!(path, cwd.join(&name));
    }

    #[tokio::test]
    async fn one_component_relative_child_publishes_complete_fork() {
        let root = directory("relative-fork-publication");
        let parent = root.join("parent");
        let kind = kind(1, false, "fixed-v1", "scripted").unwrap();
        let policy = || Increment {
            amount: 1,
            model: false,
            model_id: "fixed-v1".into(),
            provider: "scripted".into(),
        };
        {
            let trial = Trial::create(
                &parent,
                "relative-parent",
                11,
                &kind,
                Counter { total: 0 },
                policy(),
                Arc::new(ScriptedBackend),
                vec![AgentId(0)],
                Scheduling::RoundRobin,
            )
            .unwrap();
            drop(trial);
        }
        let relative_child = PathBuf::from(format!("hiyal-relative-fork-{}", std::process::id()));
        let _ = fs::remove_dir_all(&relative_child);
        let fork = fork_at_head(
            &parent,
            &relative_child,
            "test",
            "treatment",
            serde_json::json!({"delta": 2}),
            |world: &mut Counter| {
                world.total = 2;
                Ok(())
            },
            policy(),
            Arc::new(ScriptedBackend),
        )
        .unwrap();
        assert!(verify_fork::<Counter>(&relative_child).is_ok());
        drop(fork);
        fs::remove_dir_all(relative_child).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn reject_aliases_and_symlinks_into_parent_directory() {
        let root = directory("fork-containment");
        let parent = root.join("parent");
        let kind = kind(1, false, "fixed-v1", "scripted").unwrap();
        let policy = || Increment {
            amount: 1,
            model: false,
            model_id: "fixed-v1".into(),
            provider: "scripted".into(),
        };
        {
            let trial = Trial::create(
                &parent,
                "containment",
                9,
                &kind,
                Counter { total: 0 },
                policy(),
                Arc::new(ScriptedBackend),
                vec![AgentId(0)],
                Scheduling::RoundRobin,
            )
            .unwrap();
            drop(trial);
        }
        let aliased = root.join(".").join("parent").join("nested");
        assert!(
            fork_at_head(
                &parent,
                &aliased,
                "shock",
                "treatment",
                serde_json::json!({}),
                |_world: &mut Counter| Ok(()),
                policy(),
                Arc::new(ScriptedBackend)
            )
            .is_err()
        );
        assert!(!parent.join("nested").exists());
        #[cfg(unix)]
        {
            let link = root.join("source-link");
            std::os::unix::fs::symlink(&parent, &link).unwrap();
            assert!(
                fork_at_head(
                    &parent,
                    link.join("nested"),
                    "shock",
                    "treatment",
                    serde_json::json!({}),
                    |_world: &mut Counter| Ok(()),
                    policy(),
                    Arc::new(ScriptedBackend)
                )
                .is_err()
            );
            assert!(!parent.join("nested").exists());
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn fork_ids_change_with_parameters_and_committed_parent_boundary() {
        let root = directory("unique-fork-identity");
        let parent = root.join("parent");
        let kind = kind(1, false, "fixed-v1", "scripted").unwrap();
        let policy = || Increment {
            amount: 1,
            model: false,
            model_id: "fixed-v1".into(),
            provider: "scripted".into(),
        };
        {
            let trial = Trial::create(
                &parent,
                "identity-source",
                7,
                &kind,
                Counter { total: 0 },
                policy(),
                Arc::new(ScriptedBackend),
                vec![AgentId(0)],
                Scheduling::RoundRobin,
            )
            .unwrap();
            drop(trial);
        }
        let a = fork_at_head(
            &parent,
            root.join("a"),
            "shock",
            "treatment",
            serde_json::json!({"change":1}),
            |world: &mut Counter| {
                world.total += 1;
                Ok(())
            },
            policy(),
            Arc::new(ScriptedBackend),
        )
        .unwrap();
        let b = fork_at_head(
            &parent,
            root.join("b"),
            "shock",
            "treatment",
            serde_json::json!({"change":2}),
            |world: &mut Counter| {
                world.total += 2;
                Ok(())
            },
            policy(),
            Arc::new(ScriptedBackend),
        )
        .unwrap();
        assert_ne!(a.store.genesis().trial_id, b.store.genesis().trial_id);
        let first = a.store.genesis().trial_id.clone();
        drop(a);
        drop(b);
        {
            let mut source =
                Trial::resume(&parent, &kind, policy(), Arc::new(ScriptedBackend), false).unwrap();
            source.run_attempts(1).await.unwrap();
            source.checkpoint().unwrap();
        }
        let c = fork_at_head(
            &parent,
            root.join("c"),
            "shock",
            "treatment",
            serde_json::json!({"change":1}),
            |world: &mut Counter| {
                world.total += 1;
                Ok(())
            },
            policy(),
            Arc::new(ScriptedBackend),
        )
        .unwrap();
        assert_ne!(first, c.store.genesis().trial_id);
        drop(c);
        assert!(verify_fork::<Counter>(root.join("a")).is_ok());
        assert!(verify_fork::<Counter>(root.join("c")).is_ok());
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn orphan_staging_directory_does_not_block_complete_fork_publication() {
        let root = directory("orphan-staging");
        let parent = root.join("parent");
        let kind = kind(1, false, "fixed-v1", "scripted").unwrap();
        let policy = || Increment {
            amount: 1,
            model: false,
            model_id: "fixed-v1".into(),
            provider: "scripted".into(),
        };
        {
            let trial = Trial::create(
                &parent,
                "orphan",
                7,
                &kind,
                Counter { total: 0 },
                policy(),
                Arc::new(ScriptedBackend),
                vec![AgentId(0)],
                Scheduling::RoundRobin,
            )
            .unwrap();
            drop(trial);
        }
        let published = root.join("branch");
        let orphan = root.join(format!(".branch.pending-{}-old", std::process::id()));
        fs::create_dir_all(&orphan).unwrap();
        fs::write(orphan.join("manifest.json"), b"half-written").unwrap();
        let result = fork_at_head(
            &parent,
            &published,
            "change",
            "treatment",
            serde_json::json!({"delta":2}),
            |world: &mut Counter| {
                world.total = 2;
                Ok(())
            },
            policy(),
            Arc::new(ScriptedBackend),
        )
        .unwrap();
        assert!(verify_fork::<Counter>(&published).is_ok());
        assert!(orphan.exists());
        drop(result);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn reject_ambiguous_sweep_cases_before_spawn() {
        let mut plan = sweep(1);
        plan.parameter_values = vec![2, 2];
        assert!(plan.jobs().is_err());
        plan.parameter_values = vec![1, 2];
        plan.name = "../escape".into();
        assert!(plan.jobs().is_err());
        plan.name = "survey".into();
        plan.max_parallel = 0;
        assert!(plan.jobs().is_err());
    }

    #[tokio::test]
    async fn verify_fork_fails_after_metadata_or_source_tampering() {
        let root = directory("tamper");
        let parent = root.join("parent");
        let kind = kind(1, false, "fixed-v1", "scripted").unwrap();
        {
            let policy = Increment {
                amount: 1,
                model: false,
                model_id: "fixed-v1".into(),
                provider: "scripted".into(),
            };
            let mut trial = Trial::create(
                &parent,
                "fork-tamper",
                5,
                &kind,
                Counter { total: 0 },
                policy,
                Arc::new(ScriptedBackend),
                vec![AgentId(0)],
                Scheduling::RoundRobin,
            )
            .unwrap();
            trial.run_attempts(4).await.unwrap();
            trial.checkpoint().unwrap();
        }
        let branch = root.join("branch");
        let trial = fork_at_head(
            &parent,
            &branch,
            "test",
            "treatment",
            serde_json::json!({"delta":1}),
            |world: &mut Counter| {
                world.total += 1;
                Ok(())
            },
            Increment {
                amount: 1,
                model: false,
                model_id: "fixed-v1".into(),
                provider: "scripted".into(),
            },
            Arc::new(ScriptedBackend),
        )
        .unwrap();
        assert!(verify_fork::<Counter>(&branch).is_ok());
        drop(trial);
        let meta_path = branch.join("fork.json");
        let original = fs::read(&meta_path).unwrap();
        let mut record: serde_json::Value = serde_json::from_slice(&original).unwrap();
        record["parameters"]["delta"] = serde_json::json!(99);
        fs::write(&meta_path, serde_json::to_vec(&record).unwrap()).unwrap();
        assert!(
            resume_fork::<Counter, _>(
                &branch,
                Increment {
                    amount: 1,
                    model: false,
                    model_id: "fixed-v1".into(),
                    provider: "scripted".into()
                },
                Arc::new(ScriptedBackend),
                true
            )
            .is_err()
        );
        fs::write(meta_path, original).unwrap();
        assert!(verify_fork::<Counter>(&branch).is_ok());
        fs::remove_dir_all(root).unwrap();
    }
}
