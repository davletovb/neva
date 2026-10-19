//! Append-only committed transitions + rejected-attempt audit, verified replay,
//! fsynced snapshots. The complete state is stored with each accepted event;
//! reducers re-apply actions during restore to detect divergence.
use crate::core::{AgentId, World, decimal_u64};
use crate::rng::Streams;
use crate::scheduler::Schedule;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

pub const SCHEMA: u32 = 1;

/// Stable UTF-8 JSON with lexically sorted keys, including nested maps. This is
/// a documented interim profile, NOT a claim of full RFC 8785 compliance.
pub fn stable_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>, String> {
    let value = serde_json::to_value(value).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    crate::core::stable_json_bytes(&value, &mut out).map_err(|e| e.to_string())?;
    Ok(out)
}
pub fn digest<T: Serialize>(value: &T) -> Result<String, String> {
    let hash = Sha256::digest(stable_bytes(value)?);
    Ok(crate::core::hex_digest(&hash))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "W: Serialize", deserialize = "W: DeserializeOwned"))]
pub struct Genesis<W> {
    pub schema: u32,
    pub trial_id: String,
    pub initial_world: W,
    pub initial_schedule: Schedule,
    pub initial_streams: Streams,
    pub world_kind: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "W: Serialize, A: Serialize",
    deserialize = "W: DeserializeOwned, A: DeserializeOwned"
))]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum Outcome<W, A> {
    Accepted {
        action: A,
        world: W,
        state_hash: String,
    },
    Rejected {
        reason: String,
    },
    /// Persisted scheduler change; does not consume an actor turn.
    Control {
        paused: bool,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "W: Serialize, A: Serialize",
    deserialize = "W: DeserializeOwned, A: DeserializeOwned"
))]
pub struct Entry<W, A> {
    pub schema: u32,
    pub trial_id: String,
    #[serde(with = "decimal_u64")]
    pub(crate) sequence: u64,
    #[serde(with = "decimal_u64")]
    pub(crate) logical_step: u64,
    pub actor: AgentId,
    pub outcome: Outcome<W, A>,
    pub after_schedule: Schedule,
    pub after_streams: Streams,
    pub previous_hash: String,
    pub hash: String,
}

#[derive(Serialize)]
struct HashFields<'a, W, A> {
    schema: u32,
    trial_id: &'a str,
    sequence: u64,
    logical_step: u64,
    actor: AgentId,
    outcome: &'a Outcome<W, A>,
    after_schedule: &'a Schedule,
    after_streams: &'a Streams,
    previous_hash: &'a str,
}

impl<W: Serialize, A: Serialize> Entry<W, A> {
    fn compute_hash(&self) -> Result<String, String> {
        digest(&HashFields {
            schema: self.schema,
            trial_id: &self.trial_id,
            sequence: self.sequence,
            logical_step: self.logical_step,
            actor: self.actor,
            outcome: &self.outcome,
            after_schedule: &self.after_schedule,
            after_streams: &self.after_streams,
            previous_hash: &self.previous_hash,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "W: Serialize", deserialize = "W: DeserializeOwned"))]
pub struct Snapshot<W> {
    pub schema: u32,
    pub trial_id: String,
    #[serde(with = "decimal_u64")]
    pub log_bytes: u64,
    #[serde(with = "decimal_u64")]
    pub sequence: u64,
    #[serde(with = "decimal_u64")]
    pub logical_step: u64,
    pub chain_head: String,
    pub world: W,
    pub schedule: Schedule,
    pub streams: Streams,
}

#[derive(Clone)]
struct Projection<W> {
    world: W,
    schedule: Schedule,
    streams: Streams,
    sequence: u64,
    logical_step: u64,
    chain_head: String,
}

/// Verified non-mutating view: never opens a writer lock or truncates logs.
pub struct ReadOnly<W> {
    pub genesis: Genesis<W>,
    pub world: W,
    pub schedule: Schedule,
    pub streams: Streams,
    pub sequence: u64,
    pub logical_step: u64,
    pub chain_head: String,
    pub log_bytes: u64,
}

pub struct Store<W: World + Serialize + DeserializeOwned, A: Serialize + DeserializeOwned + Clone>
where
    W: World<Action = A>,
{
    pub dir: PathBuf,
    pub(crate) genesis: Genesis<W>,
    pub(crate) world: W,
    pub(crate) schedule: Schedule,
    pub(crate) streams: Streams,
    pub(crate) sequence: u64,
    pub(crate) logical_step: u64,
    pub(crate) chain_head: String,
    pub(crate) log_bytes: u64,
    poisoned: bool,
    /// Exclusive single-writer lock, removed on clean close.
    _lock: WriterLock,
    _action: std::marker::PhantomData<A>,
}

pub(crate) fn atomic_write_with(
    path: &Path,
    populate: impl FnOnce(&mut File) -> Result<(), String>,
) -> Result<(), String> {
    let temporary = path.with_extension("tmp");
    // The caller holds its directory's writer lease; this is uncommitted
    // output from a crashed prior publication, never a verified record.
    if temporary.exists() {
        fs::remove_file(&temporary).map_err(|e| e.to_string())?;
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|e| format!("write {}: {e}", temporary.display()))?;
    if let Err(error) = populate(&mut file)
        .and_then(|_| file.sync_all().map_err(|e| e.to_string()))
    {
        drop(file);
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    drop(file);
    #[cfg(not(windows))]
    let result = fs::rename(&temporary, path);
    #[cfg(windows)]
    let result = atomicwrites::replace_atomic(&temporary, path);
    if let Err(error) = result {
        let _ = fs::remove_file(&temporary);
        return Err(error.to_string());
    }
    if let Some(parent) = path.parent() {
        #[cfg(unix)]
        File::open(parent)
            .and_then(|dir| dir.sync_all())
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

pub(crate) fn save_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    atomic_write_with(path, |file| file.write_all(bytes).map_err(|e| e.to_string()))
}

/// RAII lock: even a malformed manifest/snapshot never strands a lock.
struct WriterLock {
    file: File,
    path: PathBuf,
}
impl Drop for WriterLock {
    fn drop(&mut self) {
        let _ = self.file.sync_all();
        let _ = fs::remove_file(&self.path);
    }
}
pub(crate) fn sync_trial_directory(dir: &Path) -> Result<(), String> {
    #[cfg(unix)]
    File::open(dir)
        .and_then(|f| f.sync_all())
        .map_err(|e| e.to_string())?;
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
}

fn lock_dir(dir: &Path) -> Result<WriterLock, String> {
    let path = dir.join(".writer.lock");
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|e| {
            format!("trial already open or stale lock (remove ONLY after verifying no writer): {e}")
        })?;
    Ok(WriterLock { file, path })
}

impl<W, A> Store<W, A>
where
    W: World<Action = A> + Serialize + DeserializeOwned,
    A: Serialize + DeserializeOwned + Clone,
{
    pub fn create(dir: impl AsRef<Path>, genesis: Genesis<W>) -> Result<Self, String> {
        let dir = dir.as_ref();
        if dir.exists() {
            return Err("trial path already exists".into());
        }
        fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        let lock = lock_dir(dir)?;
        if genesis.schema != SCHEMA {
            return Err("unsupported manifest schema".into());
        }
        if genesis.initial_streams.trial != genesis.trial_id {
            return Err("stream/trial mismatch".into());
        }
        genesis.initial_schedule.validate()?;
        let raw = stable_bytes(&genesis)?;
        save_atomic(&dir.join("manifest.json"), &raw)?;
        File::create(dir.join("events.jsonl"))
            .and_then(|f| f.sync_all())
            .map_err(|e| e.to_string())?;
        sync_trial_directory(dir)?;
        Ok(Self {
            dir: dir.to_path_buf(),
            world: genesis.initial_world.clone(),
            schedule: genesis.initial_schedule.clone(),
            streams: genesis.initial_streams.clone(),
            sequence: 0,
            logical_step: 0,
            chain_head: digest(&genesis)?,
            log_bytes: 0,
            poisoned: false,
            _lock: lock,
            genesis,
            _action: std::marker::PhantomData,
        })
    }

    /// Inspect the committed trial without a lock, repair, or filesystem writes.
    /// A concurrent incomplete append returns an error rather than truncating it.
    pub fn inspect_read_only(dir: impl AsRef<Path>) -> Result<ReadOnly<W>, String> {
        let dir = dir.as_ref();
        let genesis: Genesis<W> = serde_json::from_slice(
            &fs::read(dir.join("manifest.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        if genesis.schema != SCHEMA || genesis.initial_streams.trial != genesis.trial_id {
            return Err("invalid read-only manifest".into());
        }
        genesis.initial_schedule.validate()?;
        let data = fs::read(dir.join("events.jsonl")).map_err(|e| e.to_string())?;
        if !data.is_empty() && !data.ends_with(b"\n") {
            return Err("uncommitted final event record; retry inspection".into());
        }
        let snap_path = dir.join("snapshot.json");
        let snapshot: Option<Snapshot<W>> = if snap_path.exists() {
            Some(
                serde_json::from_slice(&fs::read(snap_path).map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?,
            )
        } else {
            None
        };
        let mut state = Projection {
            world: genesis.initial_world.clone(),
            schedule: genesis.initial_schedule.clone(),
            streams: genesis.initial_streams.clone(),
            sequence: 0,
            logical_step: 0,
            chain_head: digest(&genesis)?,
        };
        let mut snapshot_state = if snapshot.as_ref().is_some_and(|s| s.log_bytes == 0) {
            Some(state.clone())
        } else {
            None
        };
        let mut offset = 0u64;
        for line in data
            .split_inclusive(|b| *b == b'\n')
            .filter(|l| !l.is_empty())
        {
            let event: Entry<W, A> = serde_json::from_slice(line).map_err(|e| e.to_string())?;
            Self::validate_entry(&genesis, &mut state, &event)?;
            offset = offset
                .checked_add(line.len() as u64)
                .ok_or("offset overflow")?;
            if snapshot.as_ref().is_some_and(|s| s.log_bytes == offset) {
                snapshot_state = Some(state.clone());
            }
        }
        if let Some(snapshot) = snapshot {
            let prior = snapshot_state.ok_or("snapshot offset not on event boundary")?;
            if snapshot.schema != SCHEMA
                || snapshot.trial_id != genesis.trial_id
                || snapshot.sequence != prior.sequence
                || snapshot.logical_step != prior.logical_step
                || snapshot.chain_head != prior.chain_head
                || digest(&snapshot.world)? != digest(&prior.world)?
                || snapshot.schedule != prior.schedule
                || snapshot.streams != prior.streams
            {
                return Err("read-only snapshot validation failed".into());
            }
        }
        Ok(ReadOnly {
            genesis,
            world: state.world,
            schedule: state.schedule,
            streams: state.streams,
            sequence: state.sequence,
            logical_step: state.logical_step,
            chain_head: state.chain_head,
            log_bytes: offset,
        })
    }

    pub fn open(dir: impl AsRef<Path>) -> Result<Self, String> {
        let dir = dir.as_ref();
        let lock = lock_dir(dir)?;
        // A crashed atomic snapshot writer may leave this uncommitted file.
        // The existing writer lock guarantees there is no concurrent writer.
        let uncommitted = dir.join("snapshot.tmp");
        if uncommitted.exists() {
            fs::remove_file(&uncommitted).map_err(|e| e.to_string())?;
        }
        let raw = fs::read(dir.join("manifest.json")).map_err(|e| e.to_string())?;
        let genesis: Genesis<W> = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
        if genesis.schema != SCHEMA || genesis.initial_streams.trial != genesis.trial_id {
            return Err("unsupported manifest/trial schema".into());
        }
        genesis.initial_schedule.validate()?;
        let mut projected = Projection {
            world: genesis.initial_world.clone(),
            schedule: genesis.initial_schedule.clone(),
            streams: genesis.initial_streams.clone(),
            sequence: 0,
            logical_step: 0,
            chain_head: digest(&genesis)?,
        };
        let data = fs::read(dir.join("events.jsonl")).map_err(|e| e.to_string())?;
        let durable_length = data
            .iter()
            .rposition(|b| *b == b'\n')
            .map(|i| i + 1)
            .unwrap_or(0);
        // Read the one snapshot *before* replay. Keep at most one matching projection,
        // never a clone of every event's entire world.
        let snap_path = dir.join("snapshot.json");
        let snapshot: Option<Snapshot<W>> = if snap_path.exists() {
            Some(
                serde_json::from_slice(&fs::read(&snap_path).map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?,
            )
        } else {
            None
        };
        let initial_projection = projected.clone();
        let mut matched = if snapshot.as_ref().is_some_and(|s| s.log_bytes == 0) {
            Some(initial_projection)
        } else {
            None
        };
        let mut cursor = 0usize;
        for raw_line in data[..durable_length].split_inclusive(|b| *b == b'\n') {
            let entry: Entry<W, A> = serde_json::from_slice(raw_line)
                .map_err(|e| format!("corrupt committed log: {e}"))?;
            Self::validate_entry(&genesis, &mut projected, &entry)?;
            cursor += raw_line.len();
            if snapshot
                .as_ref()
                .is_some_and(|s| s.log_bytes == cursor as u64)
            {
                matched = Some(projected.clone());
            }
        }
        if let Some(snapshot) = snapshot {
            let expected = matched.ok_or("snapshot offset not on a committed boundary")?;
            if snapshot.schema != SCHEMA
                || snapshot.trial_id != genesis.trial_id
                || snapshot.sequence != expected.sequence
                || snapshot.logical_step != expected.logical_step
                || snapshot.chain_head != expected.chain_head
                || digest(&snapshot.world)? != digest(&expected.world)?
                || snapshot.schedule != expected.schedule
                || snapshot.streams != expected.streams
            {
                return Err("snapshot failed independent log validation".into());
            }
        }
        if durable_length < data.len() {
            // Drop only an uncommitted final fragment; all newline-terminated
            // entries have been fully validated before modifying any file.
            let f = OpenOptions::new()
                .write(true)
                .open(dir.join("events.jsonl"))
                .map_err(|e| e.to_string())?;
            f.set_len(durable_length as u64)
                .and_then(|_| f.sync_all())
                .map_err(|e| e.to_string())?;
        }
        Ok(Self {
            dir: dir.to_owned(),
            genesis,
            world: projected.world,
            schedule: projected.schedule,
            streams: projected.streams,
            sequence: projected.sequence,
            logical_step: projected.logical_step,
            chain_head: projected.chain_head,
            log_bytes: durable_length as u64,
            poisoned: false,
            _lock: lock,
            _action: std::marker::PhantomData,
        })
    }

    fn validate_entry(
        genesis: &Genesis<W>,
        state: &mut Projection<W>,
        event: &Entry<W, A>,
    ) -> Result<(), String> {
        if event.schema != SCHEMA
            || event.trial_id != genesis.trial_id
            || event.sequence != state.sequence
            || event.logical_step != state.logical_step
            || event.previous_hash != state.chain_head
            || event.compute_hash()? != event.hash
        {
            return Err("log identity/hash/sequence mismatch".into());
        }
        state.schedule.validate()?;
        let mut sched = state.schedule.clone();
        let mut rng = state.streams.clone();
        if let Outcome::Control { paused } = &event.outcome {
            if *paused {
                sched.pause(event.actor)?;
            } else {
                sched.resume(event.actor)?;
            }
            if sched != event.after_schedule || rng != event.after_streams {
                return Err("logged control change differs from scheduler state".into());
            }
            state.sequence = state.sequence.checked_add(1).ok_or("sequence overflow")?;
            state.chain_head = event.hash.clone();
            state.schedule = sched;
            return Ok(());
        }
        let selected = sched.select(&mut rng)?;
        if selected != Some(event.actor)
            || sched != event.after_schedule
            || rng != event.after_streams
        {
            return Err("log selection/RNG divergence".into());
        }
        match &event.outcome {
            Outcome::Accepted {
                action,
                world,
                state_hash,
            } => {
                let mut next = state.world.clone();
                next.apply(event.actor, action.clone())
                    .map_err(|e| format!("replay reducer failure: {e}"))?;
                if digest(&next)? != *state_hash || digest(&next)? != digest(world)? {
                    return Err("replayed world state differs from committed state".into());
                }
                state.world = next;
                state.logical_step = state.logical_step.checked_add(1).ok_or("step overflow")?;
            }
            Outcome::Rejected { .. } => {}
            Outcome::Control { .. } => unreachable!("handled before actor selection"),
        }
        state.sequence = state.sequence.checked_add(1).ok_or("sequence overflow")?;
        state.chain_head = event.hash.clone();
        state.schedule = sched;
        state.streams = rng;
        Ok(())
    }

    /// Write + fsync first, then publish the projected world/schedule in memory.
    /// On failure, the writer is poisoned to prevent appending after a torn tail.
    /// Internal staged append: replay independently re-applies actions during restore.
    pub(crate) fn append(
        &mut self,
        actor: AgentId,
        outcome: Outcome<W, A>,
        schedule: Schedule,
        streams: Streams,
    ) -> Result<Entry<W, A>, String> {
        if self.poisoned {
            return Err("writer poisoned; reopen/recover".into());
        }
        self.schedule.validate()?;
        let mut expected_schedule = self.schedule.clone();
        let mut expected_streams = self.streams.clone();
        if let Outcome::Control { paused } = &outcome {
            if *paused {
                expected_schedule.pause(actor)?;
            } else {
                expected_schedule.resume(actor)?;
            }
        } else if expected_schedule.select(&mut expected_streams)? != Some(actor) {
            return Err("unexpected selected actor".into());
        }
        if schedule != expected_schedule || streams != expected_streams {
            return Err("scheduler or random stream divergence".into());
        }
        let accepted = matches!(&outcome, Outcome::Accepted { .. });
        let next_step = self
            .logical_step
            .checked_add(u64::from(accepted))
            .ok_or("logical-step overflow")?;
        let next_sequence = self.sequence.checked_add(1).ok_or("sequence overflow")?;
        let new_world = match &outcome {
            Outcome::Accepted {
                world, state_hash, ..
            } => {
                if digest(world)? != *state_hash {
                    return Err("staged state hash mismatch".into());
                }
                world.clone()
            }
            Outcome::Rejected { .. } | Outcome::Control { .. } => self.world.clone(),
        };
        let mut event = Entry {
            schema: SCHEMA,
            trial_id: self.genesis.trial_id.clone(),
            sequence: self.sequence,
            logical_step: self.logical_step,
            actor,
            outcome,
            after_schedule: schedule,
            after_streams: streams,
            previous_hash: self.chain_head.clone(),
            hash: String::new(),
        };
        event.hash = event.compute_hash()?;
        let mut encoded = stable_bytes(&event)?;
        encoded.push(b'\n');
        let write_result = (|| -> std::io::Result<()> {
            let mut file = OpenOptions::new()
                .append(true)
                .open(self.dir.join("events.jsonl"))?;
            file.write_all(&encoded)?;
            file.sync_all()
        })();
        if let Err(error) = write_result {
            self.poisoned = true;
            return Err(format!("event append/fsync failed: {error}"));
        }
        self.log_bytes += encoded.len() as u64;
        self.sequence = next_sequence;
        self.logical_step = next_step;
        self.chain_head = event.hash.clone();
        self.world = new_world;
        self.schedule = event.after_schedule.clone();
        self.streams = event.after_streams.clone();
        Ok(event)
    }

    /// Read-only views: consumers cannot corrupt authoritative state or IDs.
    pub fn genesis(&self) -> &Genesis<W> {
        &self.genesis
    }
    pub fn world(&self) -> &W {
        &self.world
    }
    pub fn sequence(&self) -> u64 {
        self.sequence
    }
    pub fn logical_step(&self) -> u64 {
        self.logical_step
    }
    pub fn chain_head(&self) -> &str {
        &self.chain_head
    }
    pub fn log_bytes(&self) -> u64 {
        self.log_bytes
    }

    pub fn set_paused(&mut self, actor: AgentId, paused: bool) -> Result<(), String> {
        let mut scheduled = self.schedule.clone();
        if paused {
            scheduled.pause(actor)?;
        } else {
            scheduled.resume(actor)?;
        }
        self.append(
            actor,
            Outcome::Control { paused },
            scheduled,
            self.streams.clone(),
        )?;
        Ok(())
    }

    pub fn snapshot(&self) -> Result<(), String> {
        let snapshot = Snapshot {
            schema: SCHEMA,
            trial_id: self.genesis.trial_id.clone(),
            log_bytes: self.log_bytes,
            sequence: self.sequence,
            logical_step: self.logical_step,
            chain_head: self.chain_head.clone(),
            world: self.world.clone(),
            schedule: self.schedule.clone(),
            streams: self.streams.clone(),
        };
        save_atomic(&self.dir.join("snapshot.json"), &stable_bytes(&snapshot)?)
    }

    pub fn read_events(&self) -> Result<Vec<Entry<W, A>>, String> {
        let raw = fs::read(self.dir.join("events.jsonl")).map_err(|e| e.to_string())?;
        raw.split(|b| *b == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice(line).map_err(|e| e.to_string()))
            .collect()
    }
}
