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
    io::Write,
    path::Path,
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
        let mut jobs = Vec::new();
        for parameter in &self.parameter_values {
            for replicate in 0..self.replicates {
                let seed_hex = crate::core::derive_id(
                    "r2-replicate",
                    &[
                        &self.name,
                        &self.base_seed.to_string(),
                        &replicate.to_string(),
                    ],
                );
                let seed = u64::from_str_radix(&seed_hex[..16], 16).map_err(|e| e.to_string())?;
                let id = crate::core::derive_id(
                    "r2-job",
                    &[
                        &self.name,
                        &parameter.to_string(),
                        &replicate.to_string(),
                        &seed.to_string(),
                    ],
                );
                jobs.push(Job {
                    trial_name: format!("{}-p{}-r{}", self.name, parameter, replicate),
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
                    if errors.is_empty() {
                        if let Some(next) = iterator.next() {
                            let fun = action.clone();
                            active.spawn(async move {
                                let value = fun(next.clone()).await;
                                (next, value)
                            });
                        }
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
        // Validate all limits before persisting the study's immutable plan.
        let _ = self.jobs()?;
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
            let mut file = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&plan_path)
                .map_err(|e| e.to_string())?;
            file.write_all(&planned)
                .and_then(|_| file.sync_all())
                .map_err(|e| e.to_string())?;
            crate::storage::sync_trial_directory(&root)?;
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
                            || trial.store.logical_step() != trial.store.sequence()
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
        let temp = root.join("results.tmp");
        if temp.exists() {
            fs::remove_file(&temp).map_err(|e| e.to_string())?;
        }
        let mut output = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temp)
            .map_err(|e| e.to_string())?;
        output
            .write_all(csv.as_bytes())
            .and_then(|_| output.sync_all())
            .map_err(|e| e.to_string())?;
        drop(output);
        #[cfg(not(windows))]
        fs::rename(&temp, &path).map_err(|e| e.to_string())?;
        #[cfg(windows)]
        atomicwrites::replace_atomic(&temp, &path).map_err(|e| e.to_string())?;
        crate::storage::sync_trial_directory(&root)?;
        Ok(result)
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

fn copy_evidence(source: &Path, dest: &Path) -> Result<(), String> {
    fs::create_dir_all(dest).map_err(|e| e.to_string())?;
    for name in ["manifest.json", "events.jsonl"] {
        let to = dest.join(name);
        fs::copy(source.join(name), &to).map_err(|e| e.to_string())?;
        File::open(&to)
            .and_then(|f| f.sync_all())
            .map_err(|e| e.to_string())?;
    }
    crate::storage::sync_trial_directory(dest)
}

/// Verify a self-contained ancestral log and the boundary attribution for
/// the child. Neither fork depends on the *live* parent after creation.
pub fn verify_fork<W>(child: impl AsRef<Path>) -> Result<ForkMetadata, String>
where
    W: World + Serialize + DeserializeOwned,
    W::Action: Serialize + DeserializeOwned + Clone,
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
    let parent: ReadOnly<W> = Store::<W, W::Action>::inspect_read_only(&bundle)?;
    let branch: ReadOnly<W> = Store::<W, W::Action>::inspect_read_only(child)?;
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
    Ok(meta)
}

/// Create a branch at the validated *latest committed boundary*. This
/// intentionally does not support a mid-event fork. Intervention applies once
/// to genesis, and its before/after state plus parameters are recorded.
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
    W: World + Serialize + DeserializeOwned,
    W::Action: Serialize + DeserializeOwned + Clone,
    P: EffectPolicy<W>,
    F: FnOnce(&mut W) -> Result<(), String>,
{
    if !simple_name(intervention) || !simple_name(arm) {
        return Err("invalid intervention or arm identifier".into());
    }
    let parent_dir = parent_dir.as_ref();
    let child_dir = child_dir.as_ref();
    if child_dir.exists() || child_dir.starts_with(parent_dir) || parent_dir.starts_with(child_dir)
    {
        return Err("branch directory must be new and independent of parent".into());
    }
    let parent: ReadOnly<W> = Store::<W, W::Action>::inspect_read_only(parent_dir)?;
    let before = digest(&parent.world)?;
    let mut world = parent.world.clone();
    intervene(&mut world)?;
    let after = digest(&world)?;
    let name = format!("fork:{}:{intervention}:{arm}", parent.genesis.trial_id);
    let branch_id = TrialId::new(&name, parent.streams.root_seed).0;
    let streams = parent.streams.paired_branch(branch_id.clone());
    let genesis = Genesis {
        schema: SCHEMA,
        trial_id: branch_id.clone(),
        initial_world: world,
        initial_schedule: parent.schedule.clone(),
        initial_streams: streams,
        world_kind: parent.genesis.world_kind.clone(),
    };
    let trial = Trial::create_from_genesis(child_dir, genesis, policy, backend)?;
    let result = (|| -> Result<(), String> {
        let bundle = child_dir.join("ancestry");
        copy_evidence(parent_dir, &bundle)?;
        let verified: ReadOnly<W> = Store::<W, W::Action>::inspect_read_only(&bundle)?;
        if verified.chain_head != parent.chain_head
            || verified.sequence != parent.sequence
            || digest(&verified.world)? != before
        {
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
        let mut file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(child_dir.join("fork.json"))
            .map_err(|e| e.to_string())?;
        file.write_all(&stable_bytes(&meta)?)
            .and_then(|_| file.sync_all())
            .map_err(|e| e.to_string())?;
        crate::storage::sync_trial_directory(child_dir)?;
        verify_fork::<W>(child_dir)?;
        Ok(())
    })();
    if let Err(error) = result {
        drop(trial);
        let _ = fs::remove_dir_all(child_dir);
        return Err(error);
    }
    Ok(trial)
}

/// Paired branches share an independently verified parent boundary and RNG
/// stream domain, but have different trial IDs, histories, and effect ledgers.
pub fn paired_forks<W, P, F>(
    parent: impl AsRef<Path>,
    root: impl AsRef<Path>,
    intervention: &str,
    parameters: serde_json::Value,
    make_policy: impl Fn() -> P,
    treatment: F,
) -> Result<(Trial<W, P>, Trial<W, P>), String>
where
    W: World + Serialize + DeserializeOwned,
    W::Action: Serialize + DeserializeOwned + Clone,
    P: EffectPolicy<W>,
    F: FnOnce(&mut W) -> Result<(), String>,
{
    let root = root.as_ref();
    fs::create_dir_all(root).map_err(|e| e.to_string())?;
    let control = fork_at_head(
        parent.as_ref(),
        root.join("control"),
        intervention,
        "control",
        serde_json::json!({}),
        |_world: &mut W| Ok(()),
        make_policy(),
        Arc::new(ScriptedBackend),
    )?;
    let treatment = match fork_at_head(
        parent.as_ref(),
        root.join("treatment"),
        intervention,
        "treatment",
        parameters,
        treatment,
        make_policy(),
        Arc::new(ScriptedBackend),
    ) {
        Ok(value) => value,
        Err(error) => {
            drop(control);
            let _ = fs::remove_dir_all(root.join("control"));
            return Err(error);
        }
    };
    if control.store.genesis().initial_streams.seed_domain
        != treatment.store.genesis().initial_streams.seed_domain
        || control.store.genesis().initial_schedule != treatment.store.genesis().initial_schedule
        || verify_fork::<W>(root.join("control"))?.parent_head
            != verify_fork::<W>(root.join("treatment"))?.parent_head
    {
        drop(control);
        drop(treatment);
        return Err("paired branches do not share RNG/scheduler boundary".into());
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
    W: World + Serialize + DeserializeOwned,
    W::Action: Serialize + DeserializeOwned + Clone,
    P: EffectPolicy<W>,
{
    verify_fork::<W>(&dir)?;
    let genesis: Genesis<W> = serde_json::from_slice(
        &fs::read(dir.as_ref().join("manifest.json")).map_err(|e| e.to_string())?,
    )
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
