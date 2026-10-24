//! Append-only committed transitions + rejected-attempt audit, verified replay,
//! fsynced snapshots. Accepted events contain an action and world-state hash;
//! legacy events with embedded worlds remain readable and verified.
use crate::core::{AgentId, World, decimal_u64};
use crate::rng::Streams;
use crate::scheduler::Schedule;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use std::{
    any::TypeId,
    cell::RefCell,
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

pub const SCHEMA: u32 = 1;

/// Durability of the authoritative event log. `EveryEvent` is the default;
/// `AtCheckpoint` trades crash durability for throughput and must be opted
/// into explicitly by the application.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Durability {
    #[default]
    EveryEvent,
    AtCheckpoint,
}

/// Per-record limit bounds corrupted or hostile files during streaming replay.
pub const MAX_EVENT_RECORD_BYTES: usize = 32 * 1024 * 1024;
pub const MAX_METADATA_BYTES: usize = 32 * 1024 * 1024;
/// A size admission failure occurs before any append or in-memory state change.
pub(crate) const EVENT_TOO_LARGE: &str = "event record exceeds portable size limit";

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum AppendError {
    TooLarge,
    Other(String),
}
impl std::fmt::Display for AppendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooLarge => f.write_str(EVENT_TOO_LARGE),
            Self::Other(s) => f.write_str(s),
        }
    }
}
impl From<String> for AppendError {
    fn from(value: String) -> Self {
        Self::Other(value)
    }
}
impl From<&'static str> for AppendError {
    fn from(value: &'static str) -> Self {
        Self::Other(value.into())
    }
}

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
/// Written directly, byte-identical to `serde_json::to_value` followed by a
/// recursive key sort (see `canonical`).
pub fn stable_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>, String> {
    crate::canonical::to_stable_bytes(value).map_err(|e| e.to_string())
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
    /// Whether the own source checkout had tracked changes at compilation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_dirty: Option<bool>,
    /// SHA-256 of the engine checkout lockfile, not an external consumer's.
    // Keep schema-v1 wire spelling for existing genesis hashes.
    #[serde(rename = "engine_lock_sha256")]
    pub engine_checkout_lock_sha256: String,
}
impl BuildProvenance {
    pub fn current() -> Self {
        Self {
            engine_version: env!("CARGO_PKG_VERSION").into(),
            rustc_version: Some(env!("HIYAL_RUSTC_VERSION").into()),
            target: env!("HIYAL_BUILD_TARGET").into(),
            source_revision: option_env!("HIYAL_SOURCE_REVISION").map(str::to_owned),
            source_dirty: option_env!("HIYAL_SOURCE_DIRTY").map(|s| s == "true"),
            engine_checkout_lock_sha256: crate::core::hex_digest(&Sha256::digest(include_bytes!(
                "../Cargo.lock"
            ))),
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

/// Preserve the distinction between omitted world (compact events) and
/// an explicit JSON null world in historical full-world records.
#[derive(Clone, Debug, Default)]
pub enum LegacyWorld<W> {
    #[default]
    Missing,
    Present(W),
}
impl<W> LegacyWorld<W> {
    pub(crate) fn is_missing(&self) -> bool {
        matches!(self, Self::Missing)
    }
    pub fn as_ref(&self) -> Option<&W> {
        match self {
            Self::Present(world) => Some(world),
            Self::Missing => None,
        }
    }
    pub(crate) fn deserialize_present<'de, D>(d: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
        W: serde::Deserialize<'de>,
    {
        Ok(Self::Present(W::deserialize(d)?))
    }
}
impl<W: Serialize> Serialize for LegacyWorld<W> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Present(world) => world.serialize(s),
            Self::Missing => s.serialize_none(),
        }
    }
}

/// Written internally tagged by `outcome`. Read through `tagged` rather than
/// a derived deserializer, so numbers survive serde_json's
/// `arbitrary_precision` feature (#120); the format is unchanged.
#[derive(Clone, Debug, Serialize)]
#[serde(bound(serialize = "W: Serialize, A: Serialize"))]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum Outcome<W, A> {
    Accepted {
        action: A,
        /// Older schema-v1 events embed the world; modern events omit it.
        /// `Some` retains byte-for-byte legacy hash serialization.
        #[serde(
            default,
            skip_serializing_if = "LegacyWorld::is_missing",
            deserialize_with = "LegacyWorld::<W>::deserialize_present"
        )]
        world: LegacyWorld<W>,
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

/// `Outcome`'s fields and rules, externally tagged for `tagged::internally_tagged`.
#[derive(Deserialize)]
#[serde(bound(deserialize = "W: DeserializeOwned, A: DeserializeOwned"))]
#[serde(rename_all = "snake_case")]
enum OutcomeFields<W, A> {
    Accepted {
        action: A,
        #[serde(default, deserialize_with = "LegacyWorld::<W>::deserialize_present")]
        world: LegacyWorld<W>,
        state_hash: String,
        #[serde(default)]
        before_apply_streams: Option<Streams>,
    },
    Rejected {
        reason: String,
    },
    Control {
        paused: bool,
    },
}

impl<'de, W: DeserializeOwned, A: DeserializeOwned> Deserialize<'de> for Outcome<W, A> {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Ok(
            match crate::tagged::internally_tagged::<D, OutcomeFields<W, A>>(d, "outcome")? {
                OutcomeFields::Accepted {
                    action,
                    world,
                    state_hash,
                    before_apply_streams,
                } => Self::Accepted {
                    action,
                    world,
                    state_hash,
                    before_apply_streams,
                },
                OutcomeFields::Rejected { reason } => Self::Rejected { reason },
                OutcomeFields::Control { paused } => Self::Control { paused },
            },
        )
    }
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

impl<W, A> Entry<W, A> {
    /// Position of this record in the log, counting every event.
    pub fn sequence(&self) -> u64 {
        self.sequence
    }
    /// Accepted events before this one; an accepted event is applied at it.
    pub fn logical_step(&self) -> u64 {
        self.logical_step
    }
}

/// Callback of [`Store::inspect_read_only_with`].
type Visitor<'a, W, A> = dyn FnMut(&Entry<W, A>, &W) -> Result<(), String> + 'a;

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

/// Exact input bytes of one read-only replay: world/action types plus the
/// SHA-256 of the manifest, the optional snapshot and the whole event log.
#[derive(Clone, PartialEq, Eq)]
struct ReplayKey {
    types: (TypeId, TypeId),
    manifest: [u8; 32],
    snapshot: Option<[u8; 32]>,
    events: [u8; 32],
}

/// A verified view stored as detached, immutable bytes. `W: Clone` does not
/// promise a deep copy (a world may share interior-mutable state), so the
/// cache never hands out or keeps a live value: every hit rebuilds an owned
/// view and re-checks it against the digests recorded at verification.
struct CachedView {
    bytes: Vec<u8>,
    genesis_hash: String,
    world_hash: String,
}

#[derive(Serialize)]
#[serde(bound(serialize = "W: Serialize"))]
struct ViewRef<'a, W> {
    genesis: &'a Genesis<W>,
    world: &'a W,
    schedule: &'a Schedule,
    streams: &'a Streams,
    sequence: u64,
    logical_step: u64,
    chain_head: &'a str,
    log_bytes: u64,
}

#[derive(Deserialize)]
#[serde(bound(deserialize = "W: DeserializeOwned"))]
struct ViewOwned<W> {
    genesis: Genesis<W>,
    world: W,
    schedule: Schedule,
    streams: Streams,
    sequence: u64,
    logical_step: u64,
    chain_head: String,
    log_bytes: u64,
}

/// Process-local, content-addressed memo of successful read-only replays.
/// Replay is deterministic in its input bytes, so a view is reused only for
/// byte-identical evidence; any changed byte is a miss and replays in full.
/// Keys are taken from the bytes the replay actually consumed, and failures
/// are never cached. Nothing is persisted. Bounded per thread by entry count
/// and by total serialized bytes; a view larger than the budget is not kept.
const REPLAY_CACHE_LIMIT: usize = 32;
const REPLAY_CACHE_BYTES: usize = 256 * 1024 * 1024;
thread_local! {
    static REPLAY_CACHE: RefCell<Vec<(ReplayKey, CachedView)>> = const { RefCell::new(Vec::new()) };
}

fn cache_view<W: Serialize>(key: ReplayKey, view: &ReadOnly<W>) -> Result<(), String> {
    let bytes = serde_json::to_vec(&ViewRef {
        genesis: &view.genesis,
        world: &view.world,
        schedule: &view.schedule,
        streams: &view.streams,
        sequence: view.sequence,
        logical_step: view.logical_step,
        chain_head: &view.chain_head,
        log_bytes: view.log_bytes,
    })
    .map_err(|e| e.to_string())?;
    if bytes.len() > REPLAY_CACHE_BYTES {
        return Ok(());
    }
    let entry = CachedView {
        bytes,
        genesis_hash: digest(&view.genesis)?,
        world_hash: digest(&view.world)?,
    };
    REPLAY_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        cache.retain(|(k, _)| *k != key);
        let mut total: usize = cache.iter().map(|(_, v)| v.bytes.len()).sum();
        while !cache.is_empty()
            && (cache.len() >= REPLAY_CACHE_LIMIT || total + entry.bytes.len() > REPLAY_CACHE_BYTES)
        {
            total -= cache.remove(0).1.bytes.len();
        }
        cache.push((key, entry));
    });
    Ok(())
}

/// Rebuild an owned view from cached bytes; `None` (and eviction) if the
/// rebuilt genesis or world does not hash to what verification recorded.
fn cached_view<W: Serialize + DeserializeOwned>(key: &ReplayKey) -> Option<ReadOnly<W>> {
    let rebuilt = REPLAY_CACHE.with(|cache| {
        let cache = cache.borrow();
        let (_, entry) = cache.iter().find(|(k, _)| k == key)?;
        let owned: ViewOwned<W> = serde_json::from_slice(&entry.bytes).ok()?;
        let intact = digest(&owned.genesis).ok()? == entry.genesis_hash
            && digest(&owned.world).ok()? == entry.world_hash;
        Some(intact.then_some(owned))
    })?;
    let Some(owned) = rebuilt else {
        REPLAY_CACHE.with(|cache| cache.borrow_mut().retain(|(k, _)| k != key));
        return None;
    };
    Some(ReadOnly {
        genesis: owned.genesis,
        world: owned.world,
        schedule: owned.schedule,
        streams: owned.streams,
        sequence: owned.sequence,
        logical_step: owned.logical_step,
        chain_head: owned.chain_head,
        log_bytes: owned.log_bytes,
    })
}

/// Drop every memoized read-only replay on this thread.
pub fn clear_replay_cache() {
    REPLAY_CACHE.with(|cache| cache.borrow_mut().clear());
}

fn sha256_array(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn file_sha256_array(path: &Path) -> std::io::Result<[u8; 32]> {
    let mut file = File::open(path)?;
    let mut sha = Sha256::new();
    std::io::copy(&mut file, &mut sha)?;
    Ok(sha.finalize().into())
}

/// Hashes every byte replay reads, so a cache key names exactly what was verified.
struct HashingReader<R> {
    inner: R,
    sha: Option<Sha256>,
}
impl<R: Read> Read for HashingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let count = self.inner.read(buf)?;
        if let Some(sha) = &mut self.sha {
            sha.update(&buf[..count]);
        }
        Ok(count)
    }
}

#[derive(Default)]
struct ReplayInputs {
    snapshot: Option<[u8; 32]>,
    events: Option<[u8; 32]>,
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
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&path)
        .map_err(|e| e.to_string())?;
    file.try_lock()
        .map_err(|e| format!("trial already has an active writer: {e}"))?;
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
        mut inputs: Option<&mut ReplayInputs>,
        mut visit: Option<&mut Visitor<'_, W, A>>,
    ) -> Result<(Projection<W>, u64, u64), String> {
        let snap_path = dir.join("snapshot.json");
        let snapshot: Option<Snapshot<W>> = if snap_path.exists() {
            let raw = read_metadata_limited(&snap_path)?;
            if let Some(inputs) = inputs.as_deref_mut() {
                inputs.snapshot = Some(sha256_array(&raw));
            }
            Some(serde_json::from_slice(&raw).map_err(|e| e.to_string())?)
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
        if trust_snapshot && let Some(ref s) = snapshot {
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
                input
                    .seek(SeekFrom::Start(s.log_bytes - 1))
                    .map_err(|e| e.to_string())?;
                let mut delimiter = [0u8; 1];
                input
                    .read_exact(&mut delimiter)
                    .map_err(|e| e.to_string())?;
                if delimiter != *b"\n" {
                    return Err("trusted snapshot offset is not an event boundary".into());
                }
            }
            input
                .seek(SeekFrom::Start(s.log_bytes))
                .map_err(|e| e.to_string())?;
            state = Projection {
                world: s.world.clone(),
                schedule: s.schedule.clone(),
                streams: s.streams.clone(),
                sequence: s.sequence,
                logical_step: s.logical_step,
                chain_head: s.chain_head.clone(),
            };
            offset = s.log_bytes;
            matched = Some(state.clone());
        }
        // Only a full replay from byte zero yields a whole-log digest.
        let hash_events = inputs.is_some() && offset == 0;
        let mut reader = BufReader::new(HashingReader {
            inner: input,
            sha: hash_events.then(Sha256::new),
        });
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
            if let Some(visit) = visit.as_deref_mut() {
                visit(&event, &state.world)?;
            }
            offset = offset.checked_add(count as u64).ok_or("offset overflow")?;
            if snapshot.as_ref().is_some_and(|s| s.log_bytes == offset) {
                matched = Some(state.clone());
            }
        }
        let HashingReader { inner: input, sha } = reader.into_inner();
        if let (Some(inputs), Some(sha)) = (inputs, sha) {
            inputs.events = Some(sha.finalize().into());
        }
        let physical_len = input.metadata().map_err(|e| e.to_string())?.len();
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
        let raw_manifest = read_metadata_limited(&dir.join("manifest.json"))?;
        Self::inspect_manifest_bytes(dir, &raw_manifest, None, None)
    }

    /// The same full verification as [`Self::inspect_read_only`], calling
    /// `visit` with each committed event, in log order, and the world as it
    /// stands after that event (unchanged by rejected and control events).
    /// An error from `visit` stops the replay and is returned.
    ///
    /// This is how an experiment derives a trajectory from the journal in one
    /// verified pass, instead of storing its own history in the world: a
    /// world that appends a record per event pays for serializing, hashing
    /// and cloning that growing history at every step, O(T²) over a trial.
    /// `visit` sees only events that passed every hash, chain and replay
    /// check before it; the whole history is verified when this returns `Ok`.
    pub fn inspect_read_only_with(
        dir: impl AsRef<Path>,
        mut visit: impl FnMut(&Entry<W, A>, &W) -> Result<(), String>,
    ) -> Result<ReadOnly<W>, String> {
        let dir = dir.as_ref();
        let raw_manifest = read_metadata_limited(&dir.join("manifest.json"))?;
        Self::inspect_manifest_bytes(dir, &raw_manifest, None, Some(&mut visit))
    }

    /// The same full verification as [`Self::inspect_read_only`], memoized
    /// in-process by the SHA-256 of the exact manifest, snapshot and log bytes.
    /// Re-inspecting byte-identical evidence (a fork's ancestry copy of its
    /// parent, or a journal verified earlier in the same command) returns the
    /// earlier view instead of replaying again. See [`clear_replay_cache`].
    pub fn inspect_read_only_cached(dir: impl AsRef<Path>) -> Result<ReadOnly<W>, String>
    where
        W: 'static,
        A: 'static,
    {
        let dir = dir.as_ref();
        let raw_manifest = read_metadata_limited(&dir.join("manifest.json"))?;
        let types = (TypeId::of::<W>(), TypeId::of::<A>());
        let manifest = sha256_array(&raw_manifest);
        let snap_path = dir.join("snapshot.json");
        // A lookup key from the current bytes; an I/O error just skips the memo.
        let lookup = (|| -> std::io::Result<ReplayKey> {
            Ok(ReplayKey {
                types,
                manifest,
                snapshot: if snap_path.exists() {
                    Some(file_sha256_array(&snap_path)?)
                } else {
                    None
                },
                events: file_sha256_array(&dir.join("events.jsonl"))?,
            })
        })();
        if let Ok(key) = &lookup
            && let Some(view) = cached_view::<W>(key)
        {
            return Ok(view);
        }
        let mut inputs = ReplayInputs::default();
        let view = Self::inspect_manifest_bytes(dir, &raw_manifest, Some(&mut inputs), None)?;
        if let Some(events) = inputs.events {
            let key = ReplayKey {
                types,
                manifest,
                snapshot: inputs.snapshot,
                events,
            };
            cache_view(key, &view)?;
        }
        Ok(view)
    }

    fn inspect_manifest_bytes(
        dir: &Path,
        raw_manifest: &[u8],
        inputs: Option<&mut ReplayInputs>,
        visit: Option<&mut Visitor<'_, W, A>>,
    ) -> Result<ReadOnly<W>, String> {
        let genesis: Genesis<W> =
            serde_json::from_slice(raw_manifest).map_err(|e| e.to_string())?;
        if genesis.schema != SCHEMA || genesis.initial_streams.trial != genesis.trial_id {
            return Err("invalid read-only manifest".into());
        }
        genesis.initial_schedule.validate()?;
        let (state, offset, _) = Self::replay_history(dir, &genesis, false, false, inputs, visit)?;
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
        let (projected, durable_length, file_length) =
            Self::replay_history(dir, &genesis, true, trust_snapshot, None, None)?;
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
                state_hash,
                before_apply_streams,
            } => {
                // Applied in place: any error below fails the whole replay,
                // and its caller discards this projection, so a half-applied
                // world is never observed and no clone is needed.
                let next = &mut state.world;
                if let Some(before) = before_apply_streams {
                    before.checked_scope_progress(&rng, &format!("agent/{}", event.actor.0))?;
                    let mut reducer_rng = before.clone();
                    next.apply_with_streams(event.actor, action.clone(), &mut reducer_rng)
                        .map_err(|e| format!("replay reducer failure: {e}"))?;
                    reducer_rng.checked_scope_progress(before, "world")?;
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
                if digest(&*next)? != *state_hash
                    || world
                        .as_ref()
                        .is_some_and(|legacy| digest(legacy).ok().as_ref() != Some(state_hash))
                {
                    return Err("replayed world state differs from committed state".into());
                }
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
        self.append_typed(actor, outcome, schedule, streams)
            .map_err(|e| e.to_string())
    }

    pub(crate) fn append_typed(
        &mut self,
        actor: AgentId,
        outcome: Outcome<W, A>,
        schedule: Schedule,
        streams: Streams,
    ) -> Result<Entry<W, A>, AppendError> {
        self.append_inner(actor, outcome, schedule, streams, false)
    }

    /// Commit the store's own world, which the engine already staged in
    /// place by applying `action` to it with `before_apply` streams (see
    /// `World::apply_undoable`). The state hash is computed here from that
    /// world, so it matches the committed world by construction; the store
    /// does not re-apply or re-hash it. On `Err` the world stays as staged:
    /// the caller must undo the action.
    ///
    /// Fail-fast boundary: unlike `append_typed`, this does not re-run the
    /// reducer before committing, so a reducer that is not deterministic in
    /// (world, action, streams) is no longer rejected at append time. It is
    /// rejected by the next full replay (`open`, `inspect_read_only`), which
    /// re-applies every action and compares state hashes and RNG positions.
    /// Evidence should therefore be sealed only after a fresh full replay.
    pub(crate) fn commit_in_place(
        &mut self,
        actor: AgentId,
        action: A,
        before_apply: Streams,
        schedule: Schedule,
        streams: Streams,
    ) -> Result<Entry<W, A>, AppendError> {
        let state_hash = digest(&self.world)?;
        let outcome = Outcome::Accepted {
            action,
            world: LegacyWorld::Missing,
            state_hash,
            before_apply_streams: Some(before_apply),
        };
        self.append_inner(actor, outcome, schedule, streams, true)
    }

    fn append_inner(
        &mut self,
        actor: AgentId,
        outcome: Outcome<W, A>,
        schedule: Schedule,
        streams: Streams,
        staged_in_place: bool,
    ) -> Result<Entry<W, A>, AppendError> {
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
        // `None` keeps `self.world`: it is already the staged world, or the
        // event does not change it.
        let new_world = match (&outcome, staged_in_place) {
            (
                Outcome::Accepted {
                    before_apply_streams: Some(before),
                    ..
                },
                true,
            ) => {
                // Same stream-scope checks as the re-applying path; the RNG
                // equality there is implied by the caller staging with these
                // exact streams, and replay re-checks it independently.
                before.checked_scope_progress(&expected_streams, &format!("agent/{}", actor.0))?;
                streams.checked_scope_progress(before, "world")?;
                None
            }
            (_, true) => return Err("staged world requires a staged accepted outcome".into()),
            (
                Outcome::Accepted {
                    action,
                    world,
                    state_hash,
                    before_apply_streams,
                },
                false,
            ) => {
                let mut next = self.world.clone();
                if let Some(before) = before_apply_streams {
                    before
                        .checked_scope_progress(&expected_streams, &format!("agent/{}", actor.0))?;
                    let mut reducer_rng = before.clone();
                    next.apply_with_streams(actor, action.clone(), &mut reducer_rng)?;
                    reducer_rng.checked_scope_progress(before, "world")?;
                    if reducer_rng != streams {
                        return Err("staged world RNG divergence".into());
                    }
                } else {
                    if streams != expected_streams {
                        return Err("legacy RNG divergence".into());
                    }
                    next.apply(actor, action.clone())?;
                }
                if digest(&next)? != *state_hash
                    || world
                        .as_ref()
                        .is_some_and(|legacy| digest(legacy).ok().as_ref() != Some(state_hash))
                {
                    return Err("staged state hash mismatch".into());
                }
                Some(next)
            }
            (Outcome::Rejected { .. } | Outcome::Control { .. }, false) => None,
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
            return Err(AppendError::TooLarge);
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
            return Err(AppendError::Other(format!(
                "event append/fsync failed: {error}"
            )));
        }
        self.log_bytes += encoded.len() as u64;
        self.sequence = next_sequence;
        self.logical_step = next_step;
        self.chain_head = event.hash.clone();
        if let Some(world) = new_world {
            self.world = world;
        }
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

    pub fn durability(&self) -> Durability {
        self.durability
    }

    /// Flush committed bytes, including when using checkpoint-only durability.
    pub fn sync_log(&self) -> Result<(), String> {
        OpenOptions::new()
            .write(true)
            .open(self.dir.join("events.jsonl"))
            .and_then(|f| f.sync_all())
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
        self.visit_events(|entry| {
            result.push(entry);
            Ok(())
        })?;
        Ok(result)
    }

    pub fn visit_events(
        &self,
        mut visitor: impl FnMut(Entry<W, A>) -> Result<(), String>,
    ) -> Result<(), String> {
        let source = File::open(self.dir.join("events.jsonl")).map_err(|e| e.to_string())?;
        let mut reader = BufReader::new(source.take(self.log_bytes));
        let mut line = Vec::new();
        let mut consumed = 0u64;
        loop {
            let n = read_record_limited(&mut reader, &mut line)?;
            if n == 0 {
                break;
            }
            if !line.ends_with(b"\n") {
                return Err("incomplete bounded event".into());
            }
            consumed += n as u64;
            visitor(serde_json::from_slice(&line).map_err(|e| e.to_string())?)?;
        }
        if consumed != self.log_bytes {
            return Err("event log shrank during read".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod r6_legacy_world_compatibility {
    use super::*;
    #[test]
    fn historic_null_and_compact_absent_worlds_are_distinct() {
        // Serde's Option<W> loses this distinction for W whose wire value
        // is JSON null, breaking schema-v1 committed hashes on old worlds.
        let historic = r#"{"outcome":"accepted","action":1,"world":null,"state_hash":"abc"}"#;
        let parsed: Outcome<serde_json::Value, u64> = serde_json::from_str(historic).unwrap();
        let encoded = serde_json::to_value(parsed).unwrap();
        assert!(encoded.as_object().unwrap().contains_key("world"));
        assert!(encoded["world"].is_null());

        let compact = r#"{"outcome":"accepted","action":1,"state_hash":"abc"}"#;
        let parsed: Outcome<serde_json::Value, u64> = serde_json::from_str(compact).unwrap();
        let encoded = serde_json::to_value(parsed).unwrap();
        assert!(!encoded.as_object().unwrap().contains_key("world"));

        let typed: Outcome<Option<u64>, u64> = serde_json::from_str(historic).unwrap();
        let encoded = serde_json::to_value(typed).unwrap();
        assert!(encoded.as_object().unwrap().contains_key("world"));
    }
}
