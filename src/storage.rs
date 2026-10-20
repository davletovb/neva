//! Append-only committed transitions + rejected-attempt audit, verified replay,
//! fsynced snapshots. Accepted events contain an action and world-state hash;
//! legacy events with embedded worlds remain readable and verified.
use crate::core::{AgentId, World, decimal_u64};
use crate::rng::Streams;
use crate::scheduler::Schedule;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Write, Seek, SeekFrom, Read},
    path::{Path, PathBuf},
};

pub const SCHEMA: u32 = 1;

/// Durability of the authoritative event log. `EveryEvent` is the default;
/// `AtCheckpoint` trades crash durability for throughput and must be opted
/// into explicitly by the application.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Durability { EveryEvent, AtCheckpoint }

/// Per-record limit bounds corrupted or hostile files during streaming replay.
pub const MAX_EVENT_RECORD_BYTES: usize = 32 * 1024 * 1024;
pub const MAX_METADATA_BYTES: usize = 32 * 1024 * 1024;
/// A size admission failure occurs before any append or in-memory state change.
pub(crate) const EVENT_TOO_LARGE: &str = "event record exceeds portable size limit";

/// Single, bounded metadata read for library, CLI and research consumers.
pub fn read_metadata_limited(path: &Path) -> Result<Vec<u8>, String> {
    use std::io::Read;
    let file = File::open(path).map_err(|e| e.to_string())?;
    let mut bytes = Vec::new();
    file.take(MAX_METADATA_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > MAX_METADATA_BYTES {
        return Err("manifest or snapshot exceeds portable size limit".into());
    }
    Ok(bytes)
}

/// Read a single bounded newline-delimited log record. Never allocate beyond
/// MAX_EVENT_RECORD_BYTES + 1, including if the source grows concurrently.
pub fn read_record_limited(reader: &mut impl BufRead, line: &mut Vec<u8>) -> Result<usize, String> {
    use std::io::Read;
    line.clear();
    let count = reader
        .take(MAX_EVENT_RECORD_BYTES as u64 + 1)
        .read_until(b'\n', line)
        .map_err(|e| e.to_string())?;
    if count > MAX_EVENT_RECORD_BYTES {
        return Err(EVENT_TOO_LARGE.into());
    }
    Ok(count)
}

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

/// In-manifest producer identity is hash-chained through the genesis hash.
/// Optional values remain unknown unless supplied by build automation.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BuildProvenance {
    pub engine_version: String,
    pub rustc_version: Option<String>,
    pub target: String,
    pub source_revision: Option<String>,
    /// SHA-256 of the engine checkout lockfile, not an external consumer's.
    pub engine_lock_sha256: String,
}
impl BuildProvenance {
    pub fn current() -> Self {
        Self {
            engine_version: env!("CARGO_PKG_VERSION").into(),
            rustc_version: option_env!("HIYAL_RUSTC_VERSION").map(str::to_owned),
            target: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
            source_revision: option_env!("GITHUB_SHA").map(str::to_owned),
            engine_lock_sha256: crate::core::hex_digest(&Sha256::digest(include_bytes!("../Cargo.lock"))),
        }
    }
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build: Option<BuildProvenance>,
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
        /// Older schema-v1 events embed the world; modern events omit it.
        /// `Some` retains byte-for-byte legacy hash serialization.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        world: Option<W>,
        state_hash: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        before_apply_streams: Option<Streams>,
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
    durability: Durability,
    /// Exclusive OS-backed single-writer lease released by the kernel.
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
    if let Err(error) = populate(&mut file).and_then(|_| file.sync_all().map_err(|e| e.to_string()))
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
    atomic_write_with(path, |file| {
        file.write_all(bytes).map_err(|e| e.to_string())
    })
}

/// OS-backed lock survives stale marker paths and automatically releases
/// on process death. The marker is deliberately *never removed* to avoid
/// a race where an unlinked locked inode admits another writer.
struct WriterLock {
    _file: File,
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
    let file = OpenOptions::new().read(true).write(true).create(true)
        .open(&path).map_err(|e| e.to_string())?;
    file.try_lock().map_err(|e| format!("trial already has an active writer: {e}"))?;
    Ok(WriterLock { _file: file })
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
        // Preflight all fallible pure checks before allocating a trial path.
        if genesis.schema != SCHEMA {
            return Err("unsupported manifest schema".into());
        }
        if genesis.initial_streams.trial != genesis.trial_id {
            return Err("stream/trial mismatch".into());
        }
        genesis.initial_schedule.validate()?;
        let raw = stable_bytes(&genesis)?;
        if raw.len() > MAX_METADATA_BYTES {
            return Err("manifest exceeds portable size limit".into());
        }
        let initial_hash = digest(&genesis)?;
        fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        let lock = lock_dir(dir)?;
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
            chain_head: initial_hash,
            log_bytes: 0,
            poisoned: false,
            durability: Durability::EveryEvent,
            _lock: lock,
            genesis,
            _action: std::marker::PhantomData,
        })
    }

    /// Both mutating recovery and read-only inspection must validate exactly
    /// the same history, snapshot boundary, hashes and world transitions.
    fn replay_history(
        dir: &Path,
        genesis: &Genesis<W>,
        allow_torn_tail: bool,
        trust_snapshot: bool,
    ) -> Result<(Projection<W>, u64, u64), String> {
        let snap_path = dir.join("snapshot.json");
        let snapshot: Option<Snapshot<W>> = if snap_path.exists() {
            Some(
                serde_json::from_slice(&read_metadata_limited(&snap_path)?)
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
            chain_head: digest(genesis)?,
        };
        let mut matched = if snapshot.as_ref().is_some_and(|s| s.log_bytes == 0) {
            Some(state.clone())
        } else {
            None
        };
        let mut input = File::open(dir.join("events.jsonl")).map_err(|e| e.to_string())?;
        let mut offset = 0u64;
        // Explicit trusted-checkpoint mode is faster, but it does NOT
        // independently verify events before the snapshot. Normal open() and
        // read-only validate() still independently verify the full history.
        if trust_snapshot {
            if let Some(ref s) = snapshot {
                let file_len = input.metadata().map_err(|e| e.to_string())?.len();
                if s.schema != genesis.schema
                    || s.trial_id != genesis.trial_id
                    || s.streams.trial != genesis.trial_id
                    || s.sequence < s.logical_step
                    || s.log_bytes > file_len
                    || s.chain_head.len() != 64
                    || !s.chain_head.bytes().all(|b| b.is_ascii_hexdigit())
                {
                    return Err("invalid trusted snapshot identity/bounds".into());
                }
                s.schedule.validate()?;
                if s.log_bytes > 0 {
                    input.seek(SeekFrom::Start(s.log_bytes - 1))
                        .map_err(|e| e.to_string())?;
                    let mut delimiter = [0u8; 1];
                    input.read_exact(&mut delimiter).map_err(|e| e.to_string())?;
                    if delimiter != [b'\n'] {
                        return Err("trusted snapshot offset is not an event boundary".into());
                    }
                }
                input.seek(SeekFrom::Start(s.log_bytes)).map_err(|e| e.to_string())?;
                state = Projection {
                    world: s.world.clone(), schedule: s.schedule.clone(),
                    streams: s.streams.clone(), sequence: s.sequence,
                    logical_step: s.logical_step, chain_head: s.chain_head.clone(),
                };
                offset = s.log_bytes;
                matched = Some(state.clone());
            }
        }
        let mut reader = BufReader::new(input);
        let mut line = Vec::new();
        loop {
            let count = read_record_limited(&mut reader, &mut line)?;
            if count == 0 {
                break;
            }
            if !line.ends_with(b"\n") {
                if allow_torn_tail {
                    break;
                }
                return Err("uncommitted final event record; retry inspection".into());
            }
            let event: Entry<W, A> =
                serde_json::from_slice(&line).map_err(|e| format!("corrupt committed log: {e}"))?;
            Self::validate_entry(genesis, &mut state, &event)?;
            offset = offset.checked_add(count as u64).ok_or("offset overflow")?;
            if snapshot.as_ref().is_some_and(|s| s.log_bytes == offset) {
                matched = Some(state.clone());
            }
        }
        let physical_len = reader
            .get_ref()
            .metadata()
            .map_err(|e| e.to_string())?
            .len();
        if let Some(snapshot) = snapshot {
            let expected = matched.ok_or("snapshot offset not on a committed event boundary")?;
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
        Ok((state, offset, physical_len))
    }

    /// Inspect without a lock, repair, or filesystem writes. A concurrent
    /// unfinished append is not silently promoted to a committed event.
    pub fn inspect_read_only(dir: impl AsRef<Path>) -> Result<ReadOnly<W>, String> {
        let dir = dir.as_ref();
        let genesis: Genesis<W> =
            serde_json::from_slice(&read_metadata_limited(&dir.join("manifest.json"))?)
                .map_err(|e| e.to_string())?;
        if genesis.schema != SCHEMA || genesis.initial_streams.trial != genesis.trial_id {
            return Err("invalid read-only manifest".into());
        }
        genesis.initial_schedule.validate()?;
        let (state, offset, _) = Self::replay_history(dir, &genesis, false, false)?;
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
        Self::open_mode(dir, false)
    }

    /// Faster resume from a caller-*trusted* snapshot. Only verifies the
    /// suffix after the snapshot. Use inspect_read_only for full validation.
    pub fn open_trusted(dir: impl AsRef<Path>) -> Result<Self, String> {
        Self::open_mode(dir, true)
    }

    fn open_mode(dir: impl AsRef<Path>, trust_snapshot: bool) -> Result<Self, String> {
        let dir = dir.as_ref();
        let lock = lock_dir(dir)?;
        // The exclusive writer lease allows discarding an abandoned
        // temporary snapshot, not rewriting a corrupt committed record.
        let uncommitted = dir.join("snapshot.tmp");
        if uncommitted.exists() {
            fs::remove_file(&uncommitted).map_err(|e| e.to_string())?;
        }
        let genesis: Genesis<W> =
            serde_json::from_slice(&read_metadata_limited(&dir.join("manifest.json"))?)
                .map_err(|e| e.to_string())?;
        if genesis.schema != SCHEMA || genesis.initial_streams.trial != genesis.trial_id {
            return Err("unsupported manifest/trial schema".into());
        }
        genesis.initial_schedule.validate()?;
        let (projected, durable_length, file_length) = Self::replay_history(dir, &genesis, true, trust_snapshot)?;
        if durable_length < file_length {
            let f = OpenOptions::new()
                .write(true)
                .open(dir.join("events.jsonl"))
                .map_err(|e| e.to_string())?;
            f.set_len(durable_length)
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
            log_bytes: durable_length,
            poisoned: false,
            durability: Durability::EveryEvent,
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
        if selected != Some(event.actor) || sched != event.after_schedule {
            return Err("log selection/RNG divergence".into());
        }
        event.after_streams.checked_progress(&rng)?;
        match &event.outcome {
            Outcome::Accepted {
                action,
                world,
                state_hash, before_apply_streams,
            } => {
                let mut next = state.world.clone();
                if let Some(before) = before_apply_streams {
                    before.checked_progress(&rng)?;
                    let mut reducer_rng = before.clone();
                    next.apply_with_streams(event.actor, action.clone(), &mut reducer_rng)
                        .map_err(|e| format!("replay reducer failure: {e}"))?;
                    if reducer_rng != event.after_streams {
                        return Err("replayed world RNG diverges".into());
                    }
                } else {
                    // Legacy event: no stochastic reducer channel.
                    if rng != event.after_streams {
                        return Err("legacy RNG stream divergence".into());
                    }
                    next.apply(event.actor, action.clone())
                        .map_err(|e| format!("replay reducer failure: {e}"))?;
                }
                if digest(&next)? != *state_hash
                    || world.as_ref().is_some_and(|legacy| digest(legacy).ok().as_ref() != Some(state_hash))
                {
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
        state.streams = event.after_streams.clone();
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
        if schedule != expected_schedule {
            return Err("scheduler divergence".into());
        }
        streams.checked_progress(&expected_streams)?;
        let accepted = matches!(&outcome, Outcome::Accepted { .. });
        let next_step = self
            .logical_step
            .checked_add(u64::from(accepted))
            .ok_or("logical-step overflow")?;
        let next_sequence = self.sequence.checked_add(1).ok_or("sequence overflow")?;
        let new_world = match &outcome {
            Outcome::Accepted { action, world, state_hash, before_apply_streams } => {
                let mut next = self.world.clone();
                if let Some(before) = before_apply_streams {
                    before.checked_progress(&expected_streams)?;
                    let mut reducer_rng = before.clone();
                    next.apply_with_streams(actor, action.clone(), &mut reducer_rng)?;
                    if reducer_rng != streams {
                        return Err("staged world RNG divergence".into());
                    }
                } else {
                    if streams != expected_streams { return Err("legacy RNG divergence".into()); }
                    next.apply(actor, action.clone())?;
                }
                if digest(&next)? != *state_hash
                    || world.as_ref().is_some_and(|legacy| digest(legacy).ok().as_ref() != Some(state_hash))
                {
                    return Err("staged state hash mismatch".into());
                }
                next
            }
            Outcome::Rejected { .. } | Outcome::Control { .. } => self.world.clone(),
        };
        let mut event = Entry {
            schema: self.genesis.schema,
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
        if encoded.len() > MAX_EVENT_RECORD_BYTES {
            return Err(EVENT_TOO_LARGE.into());
        }
        let write_result = (|| -> std::io::Result<()> {
            let mut file = OpenOptions::new()
                .append(true)
                .open(self.dir.join("events.jsonl"))?;
            file.write_all(&encoded)?;
            if self.durability == Durability::EveryEvent {
                file.sync_all()?;
            }
            Ok(())
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

    /// Commit a scheduler control record and return the durable log entry.
    pub fn set_paused_event(
        &mut self,
        actor: AgentId,
        paused: bool,
    ) -> Result<Entry<W, A>, String> {
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
        )
    }

    /// Backward-compatible convenience wrapper for scheduling controls.
    pub fn set_paused(&mut self, actor: AgentId, paused: bool) -> Result<(), String> {
        self.set_paused_event(actor, paused).map(|_| ())
    }

    /// Choose a durability tradeoff. The setting affects subsequent events.
    /// Before changing modes, flush earlier writes to establish a boundary.
    pub fn set_durability(&mut self, durability: Durability) -> Result<(), String> {
        self.sync_log()?;
        self.durability = durability;
        Ok(())
    }

    pub fn durability(&self) -> Durability { self.durability }

    /// Flush committed bytes, including when using checkpoint-only durability.
    pub fn sync_log(&self) -> Result<(), String> {
        File::open(self.dir.join("events.jsonl")).and_then(|f| f.sync_all())
            .map_err(|e| e.to_string())
    }

    pub fn snapshot(&self) -> Result<(), String> {
        self.sync_log()?;
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
        let bytes = stable_bytes(&snapshot)?;
        if bytes.len() > MAX_METADATA_BYTES {
            return Err("snapshot exceeds portable size limit".into());
        }
        save_atomic(&self.dir.join("snapshot.json"), &bytes)
    }

    pub fn read_events(&self) -> Result<Vec<Entry<W, A>>, String> {
        let mut result = Vec::new();
        // Historical convenience method: reject huge outputs instead of
        // unboundedly collecting the whole log into process memory.
        if self.log_bytes > 64 * 1024 * 1024 {
            return Err("read_events memory budget exceeded; use visit_events".into());
        }
        self.visit_events(|entry| { result.push(entry); Ok(()) })?;
        Ok(result)
    }

    pub fn visit_events(&self, mut visitor: impl FnMut(Entry<W, A>) -> Result<(), String>) -> Result<(), String> {
        let source = File::open(self.dir.join("events.jsonl")).map_err(|e| e.to_string())?;
        let mut reader = BufReader::new(source.take(self.log_bytes));
        let mut line = Vec::new();
        let mut consumed = 0u64;
        loop {
            let n = read_record_limited(&mut reader, &mut line)?;
            if n == 0 { break; }
            if !line.ends_with(b"\n") { return Err("incomplete bounded event".into()); }
            consumed += n as u64;
            visitor(serde_json::from_slice(&line).map_err(|e| e.to_string())?)?;
        }
        if consumed != self.log_bytes { return Err("event log shrank during read".into()); }
        Ok(())
    }
}
