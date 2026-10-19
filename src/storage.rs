//! Append-only committed transitions + rejected-attempt audit, verified replay,
//! fsynced snapshots. The complete state is stored with each accepted event;
//! reducers re-apply actions during restore to detect divergence.
use crate::core::{AgentId, World, decimal_u64};
use crate::rng::Streams;
use crate::scheduler::Schedule;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use std::{fs::{self, File, OpenOptions}, io::{Read, Write, Seek, SeekFrom}, path::{Path, PathBuf}};

pub const SCHEMA: u32 = 1;

/// Stable UTF-8 JSON with lexically sorted keys, including nested maps. This is
/// a documented interim profile, NOT a claim of full RFC 8785 compliance.
pub fn stable_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>, String> {
    fn encode(value: &serde_json::Value, out: &mut Vec<u8>) -> serde_json::Result<()> {
        match value {
            serde_json::Value::Object(map) => {
                out.push(b'{');
                let mut keys: Vec<_> = map.keys().collect();
                keys.sort();
                for (index, key) in keys.iter().enumerate() {
                    if index != 0 { out.push(b','); }
                    serde_json::to_writer(&mut *out, key)?;
                    out.push(b':');
                    encode(&map[key.as_str()], out)?;
                }
                out.push(b'}');
            }
            serde_json::Value::Array(values) => {
                out.push(b'[');
                for (index, value) in values.iter().enumerate() {
                    if index != 0 { out.push(b','); }
                    encode(value, out)?;
                }
                out.push(b']');
            }
            scalar => serde_json::to_writer(out, scalar)?,
        }
        Ok(())
    }
    let value = serde_json::to_value(value).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    encode(&value, &mut out).map_err(|e| e.to_string())?;
    Ok(out)
}

pub fn digest<T: Serialize>(value: &T) -> Result<String, String> {
    use std::fmt::Write as _;
    let hash = Sha256::digest(stable_bytes(value)?);
    let mut out = String::new();
    for byte in hash { write!(out, "{byte:02x}").map_err(|e| e.to_string())?; }
    Ok(out)
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
#[serde(bound(serialize = "W: Serialize, A: Serialize", deserialize = "W: DeserializeOwned, A: DeserializeOwned"))]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum Outcome<W, A> {
    Accepted { action: A, world: W, state_hash: String },
    Rejected { reason: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "W: Serialize, A: Serialize", deserialize = "W: DeserializeOwned, A: DeserializeOwned"))]
pub struct Entry<W, A> {
    pub schema: u32,
    pub trial_id: String,
    #[serde(with = "decimal_u64")]
    pub sequence: u64,
    #[serde(with = "decimal_u64")]
    pub logical_step: u64,
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
    outcome: &'a Outcome<W,A>,
    after_schedule: &'a Schedule,
    after_streams: &'a Streams,
    previous_hash: &'a str,
}

impl<W: Serialize, A: Serialize> Entry<W,A> {
    fn compute_hash(&self) -> Result<String,String> {
        digest(&HashFields {
            schema: self.schema, trial_id: &self.trial_id,
            sequence: self.sequence, logical_step: self.logical_step,
            actor: self.actor, outcome: &self.outcome,
            after_schedule: &self.after_schedule, after_streams: &self.after_streams,
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

pub struct Store<W: World + Serialize + DeserializeOwned, A: Serialize + DeserializeOwned + Clone>
where W: World<Action=A> {
    pub dir: PathBuf,
    pub genesis: Genesis<W>,
    pub world: W,
    pub schedule: Schedule,
    pub streams: Streams,
    pub sequence: u64,
    pub logical_step: u64,
    pub chain_head: String,
    pub log_bytes: u64,
    poisoned: bool,
    /// Exclusive single-writer lock, removed on clean close.
    lock: File,
    _action: std::marker::PhantomData<A>,
}

fn save_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let temporary = path.with_extension("tmp");
    let mut file = OpenOptions::new().write(true).create_new(true).open(&temporary)
        .map_err(|e| format!("write {}: {e}", temporary.display()))?;
    file.write_all(bytes).map_err(|e| e.to_string())?;
    file.sync_all().map_err(|e| e.to_string())?;
    drop(file);
    fs::rename(&temporary, path).map_err(|e| e.to_string())?;
    if let Some(parent) = path.parent() {
        #[cfg(unix)]
        File::open(parent).and_then(|dir| dir.sync_all()).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn lock_dir(dir: &Path) -> Result<File,String> {
    OpenOptions::new().write(true).create_new(true).open(dir.join(".writer.lock"))
        .map_err(|e| format!("trial already open or stale lock (remove ONLY after verifying no writer): {e}"))
}

impl<W, A> Store<W, A>
where W: World<Action=A> + Serialize + DeserializeOwned, A: Serialize + DeserializeOwned + Clone {
    pub fn create(dir: impl AsRef<Path>, genesis: Genesis<W>) -> Result<Self,String> {
        let dir = dir.as_ref();
        if dir.exists() { return Err("trial path already exists".into()); }
        fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        let lock = lock_dir(dir)?;
        if genesis.schema != SCHEMA { return Err("unsupported manifest schema".into()); }
        if genesis.initial_streams.trial != genesis.trial_id { return Err("stream/trial mismatch".into()); }
        let raw = stable_bytes(&genesis)?;
        save_atomic(&dir.join("manifest.json"), &raw)?;
        File::create(dir.join("events.jsonl")).and_then(|f| f.sync_all()).map_err(|e| e.to_string())?;
        Ok(Self {
            dir: dir.to_path_buf(), world: genesis.initial_world.clone(),
            schedule: genesis.initial_schedule.clone(), streams: genesis.initial_streams.clone(),
            sequence: 0, logical_step: 0, chain_head: digest(&genesis)?,
            log_bytes: 0, poisoned: false, lock, genesis, _action: std::marker::PhantomData,
        })
    }

    pub fn open(dir: impl AsRef<Path>) -> Result<Self,String> {
        let dir = dir.as_ref();
        let lock = lock_dir(dir)?;
        let raw = fs::read(dir.join("manifest.json")).map_err(|e| e.to_string())?;
        let genesis: Genesis<W> = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
        if genesis.schema != SCHEMA || genesis.initial_streams.trial != genesis.trial_id {
            return Err("unsupported manifest/trial schema".into());
        }
        let mut projected = Projection {
            world: genesis.initial_world.clone(), schedule: genesis.initial_schedule.clone(),
            streams: genesis.initial_streams.clone(), sequence: 0,
            logical_step: 0, chain_head: digest(&genesis)?,
        };
        let data = fs::read(dir.join("events.jsonl")).map_err(|e| e.to_string())?;
        let durable_length = data.iter().rposition(|b| *b == b'\n').map(|i| i+1).unwrap_or(0);
        let mut checkpoints = Vec::new();
        let mut cursor = 0usize;
        for raw_line in data[..durable_length].split_inclusive(|b| *b == b'\n') {
            let entry: Entry<W,A> = serde_json::from_slice(raw_line).map_err(|e| format!("corrupt committed log: {e}"))?;
            Self::validate_entry(&genesis, &mut projected, &entry)?;
            cursor += raw_line.len();
            checkpoints.push((cursor as u64, projected.clone()));
        }
        // Compare snapshot against a fully independently replayed prefix.
        let snap_path = dir.join("snapshot.json");
        if snap_path.exists() {
            let raw = fs::read(&snap_path).map_err(|e| e.to_string())?;
            let snapshot: Snapshot<W> = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
            let expected = if snapshot.log_bytes == 0 {
                Projection {
                    world: genesis.initial_world.clone(), schedule: genesis.initial_schedule.clone(),
                    streams: genesis.initial_streams.clone(), sequence: 0,
                    logical_step: 0, chain_head: digest(&genesis)?,
                }
            } else {
                checkpoints.iter().find(|(offset, _)| *offset == snapshot.log_bytes)
                    .map(|(_, projection)| projection.clone())
                    .ok_or("snapshot offset not on a committed boundary")?
            };
            if snapshot.schema != SCHEMA || snapshot.trial_id != genesis.trial_id
                || snapshot.sequence != expected.sequence
                || snapshot.logical_step != expected.logical_step
                || snapshot.chain_head != expected.chain_head
                || digest(&snapshot.world)? != digest(&expected.world)?
                || snapshot.schedule != expected.schedule
                || snapshot.streams != expected.streams {
                return Err("snapshot failed independent log validation".into());
            }
        }
        if durable_length < data.len() {
            // Drop only an uncommitted final fragment; all newline-terminated
            // entries have been fully validated before modifying any file.
            let f = OpenOptions::new().write(true).open(dir.join("events.jsonl"))
                .map_err(|e| e.to_string())?;
            f.set_len(durable_length as u64).and_then(|_| f.sync_all()).map_err(|e| e.to_string())?;
        }
        Ok(Self {
            dir: dir.to_owned(), genesis, world: projected.world,
            schedule: projected.schedule, streams: projected.streams,
            sequence: projected.sequence, logical_step: projected.logical_step,
            chain_head: projected.chain_head, log_bytes: durable_length as u64,
            poisoned: false, lock, _action: std::marker::PhantomData,
        })
    }

    fn validate_entry(genesis: &Genesis<W>, state: &mut Projection<W>, event: &Entry<W,A>) -> Result<(),String> {
        if event.schema != SCHEMA || event.trial_id != genesis.trial_id
            || event.sequence != state.sequence || event.logical_step != state.logical_step
            || event.previous_hash != state.chain_head
            || event.compute_hash()? != event.hash {
            return Err("log identity/hash/sequence mismatch".into());
        }
        let mut sched = state.schedule.clone();
        let mut rng = state.streams.clone();
        let selected = sched.select(&mut rng)?;
        if selected != Some(event.actor) || sched != event.after_schedule || rng != event.after_streams {
            return Err("log selection/RNG divergence".into());
        }
        match &event.outcome {
            Outcome::Accepted {action, world, state_hash} => {
                let mut next = state.world.clone();
                next.apply(event.actor, action.clone()).map_err(|e| format!("replay reducer failure: {e}"))?;
                if digest(&next)? != *state_hash || digest(&next)? != digest(world)? {
                    return Err("replayed world state differs from committed state".into());
                }
                state.world = next;
                state.logical_step = state.logical_step.checked_add(1).ok_or("step overflow")?;
            }
            Outcome::Rejected {..} => {}
        }
        state.sequence = state.sequence.checked_add(1).ok_or("sequence overflow")?;
        state.chain_head = event.hash.clone();
        state.schedule = sched;
        state.streams = rng;
        Ok(())
    }

    /// Write + fsync first, then publish the projected world/schedule in memory.
    /// On failure, the writer is poisoned to prevent appending after a torn tail.
    pub fn append(&mut self, actor: AgentId, outcome: Outcome<W,A>, schedule: Schedule, streams: Streams) -> Result<Entry<W,A>,String> {
        if self.poisoned { return Err("writer poisoned; reopen/recover".into()); }
        let mut proposed = Projection {
            world: self.world.clone(), schedule: self.schedule.clone(),
            streams: self.streams.clone(), sequence: self.sequence,
            logical_step: self.logical_step, chain_head: self.chain_head.clone(),
        };
        let mut event = Entry {
            schema: SCHEMA, trial_id: self.genesis.trial_id.clone(), sequence: self.sequence,
            logical_step: self.logical_step, actor, outcome,
            after_schedule: schedule, after_streams: streams,
            previous_hash: self.chain_head.clone(), hash: String::new(),
        };
        event.hash = event.compute_hash()?;
        Self::validate_entry(&self.genesis, &mut proposed, &event)?;
        let mut encoded = stable_bytes(&event)?;
        encoded.push(b'\n');
        let write_result = (|| -> std::io::Result<()> {
            let mut file = OpenOptions::new().append(true).open(self.dir.join("events.jsonl"))?;
            file.write_all(&encoded)?;
            file.sync_all()
        })();
        if let Err(error) = write_result {
            self.poisoned = true;
            return Err(format!("event append/fsync failed: {error}"));
        }
        self.log_bytes += encoded.len() as u64;
        self.sequence = proposed.sequence;
        self.logical_step = proposed.logical_step;
        self.chain_head = proposed.chain_head;
        self.world = proposed.world;
        self.schedule = proposed.schedule;
        self.streams = proposed.streams;
        Ok(event)
    }

    pub fn snapshot(&self) -> Result<(),String> {
        let snapshot = Snapshot {
            schema: SCHEMA, trial_id: self.genesis.trial_id.clone(),
            log_bytes: self.log_bytes, sequence: self.sequence,
            logical_step: self.logical_step, chain_head: self.chain_head.clone(),
            world: self.world.clone(), schedule: self.schedule.clone(),
            streams: self.streams.clone(),
        };
        save_atomic(&self.dir.join("snapshot.json"), &stable_bytes(&snapshot)?)
    }

    pub fn read_events(&self) -> Result<Vec<Entry<W,A>>,String> {
        let raw = fs::read(self.dir.join("events.jsonl")).map_err(|e| e.to_string())?;
        raw.split(|b| *b==b'\n').filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice(line).map_err(|e| e.to_string())).collect()
    }
}
impl<W: World + Serialize + DeserializeOwned, A: Serialize + DeserializeOwned + Clone> Drop for Store<W,A>
where W: World<Action=A> {
    fn drop(&mut self) {
        let _ = self.lock.sync_all();
        let _ = fs::remove_file(self.dir.join(".writer.lock"));
    }
}
