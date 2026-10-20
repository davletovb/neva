//! Privacy-safe structured event projection. This non-authoritative JSONL
//! stream contains no actions, world snapshots, messages or prompt bodies.
use crate::{
    core::World,
    experiment::Observer,
    storage::{Entry, Outcome, Store, atomic_write_with, read_record_limited},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    fs::{File, OpenOptions},
    io::{BufReader, Read, Seek, SeekFrom, Write},
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::Mutex,
};

pub const OBSERVATION_SCHEMA: u32 = 1;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventSummary {
    pub schema: u32,
    pub trial_id: String,
    #[serde(with = "crate::core::decimal_u64")]
    pub sequence: u64,
    #[serde(with = "crate::core::decimal_u64")]
    pub logical_step: u64,
    pub actor: crate::core::AgentId,
    pub kind: String,
    pub event_hash: String,
    pub state_hash: Option<String>,
    /// End offset in authoritative events.jsonl, when emitted by a newer
    /// recorder. This is *derived metadata*, not a substitute for the hash.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "crate::core::decimal_u64::option"
    )]
    pub source_log_bytes: Option<u64>,
}

impl EventSummary {
    pub fn from_entry<W, A>(entry: &Entry<W, A>) -> Self {
        let (kind, state_hash) = match &entry.outcome {
            Outcome::Accepted { state_hash, .. } => ("accepted", Some(state_hash.clone())),
            Outcome::Rejected { .. } => ("rejected", None),
            Outcome::Control { .. } => ("control", None),
        };
        Self {
            schema: OBSERVATION_SCHEMA,
            trial_id: entry.trial_id.clone(),
            sequence: entry.sequence,
            logical_step: entry.logical_step,
            actor: entry.actor,
            kind: kind.to_owned(),
            event_hash: entry.hash.clone(),
            state_hash,
            source_log_bytes: None,
        }
    }
}

/// Project exactly the already-verified committed prefix, even if a writer
/// appends additional events after inspection. One writer for CLI and recovery
/// prevents drift in delimiters, blank-line handling and JSON shape.
pub fn write_summaries<W>(
    dir: &Path,
    verified: &crate::storage::ReadOnly<W>,
    out: &mut impl Write,
) -> Result<(), String>
where
    W: World + Serialize + DeserializeOwned,
    W::Action: Serialize + DeserializeOwned + Clone,
{
    let source = File::open(dir.join("events.jsonl")).map_err(|e| e.to_string())?;
    let mut reader = BufReader::new(source.take(verified.log_bytes));
    let mut line = Vec::new();
    let mut read_bytes = 0u64;
    loop {
        let count = read_record_limited(&mut reader, &mut line)?;
        if count == 0 {
            break;
        }
        read_bytes += count as u64;
        if !line.ends_with(b"\n") {
            return Err("export source has an incomplete event record".into());
        }
        let entry: Entry<W, W::Action> =
            serde_json::from_slice(&line).map_err(|e| e.to_string())?;
        let mut summary = EventSummary::from_entry(&entry);
        summary.source_log_bytes = Some(read_bytes);
        serde_json::to_writer(&mut *out, &summary).map_err(|e| e.to_string())?;
        out.write_all(b"\n").map_err(|e| e.to_string())?;
    }
    if read_bytes != verified.log_bytes {
        return Err("export source changed after verification".into());
    }
    Ok(())
}

pub struct JsonlObserver<W: World> {
    file: Mutex<File>,
    _world: PhantomData<fn() -> W>,
}

impl<W> JsonlObserver<W>
where
    W: World + Serialize + DeserializeOwned,
    W::Action: Serialize + DeserializeOwned + Clone,
{
    pub fn path(dir: impl AsRef<Path>) -> PathBuf {
        dir.as_ref().join("observations.jsonl")
    }

    /// Opt-in projection fast path: if the final already-written summary has
    /// the exact trusted Store head, keep it without scanning the full log.
    /// If the projection is incomplete, missing or corrupt, rebuild via the
    /// normal independently verified recovery path instead. The caller must
    /// have explicitly accepted trusted-snapshot semantics.
    pub fn recover_trusted(store: &Store<W, W::Action>) -> Result<Self, String> {
        let path = Self::path(&store.dir);
        if let Ok(mut file) = File::open(&path) {
            let len = file.metadata().map_err(|e| e.to_string())?.len();
            if len == 0 && store.sequence() == 0 {
                return Ok(Self {
                    file: Mutex::new(
                        OpenOptions::new()
                            .append(true)
                            .open(&path)
                            .map_err(|e| e.to_string())?,
                    ),
                    _world: PhantomData,
                });
            }
            if len > 0 && store.sequence() > 0 {
                let count = len.min(4096);
                file.seek(SeekFrom::Start(len - count))
                    .map_err(|e| e.to_string())?;
                let mut tail = vec![0u8; count as usize];
                file.read_exact(&mut tail).map_err(|e| e.to_string())?;
                if tail.last() == Some(&b'\n') {
                    let records: Vec<_> = tail
                        .split(|b| *b == b'\n')
                        .filter(|part| !part.is_empty())
                        .collect();
                    if let Some(last) = records.last()
                        && let Ok(summary) = serde_json::from_slice::<EventSummary>(last)
                        && summary.trial_id == store.genesis().trial_id
                        && summary.sequence < store.sequence()
                    {
                        if summary.sequence.checked_add(1) == Some(store.sequence())
                            && summary.event_hash == store.chain_head()
                        {
                            return Ok(Self {
                                file: Mutex::new(
                                    OpenOptions::new()
                                        .append(true)
                                        .open(&path)
                                        .map_err(|e| e.to_string())?,
                                ),
                                _world: PhantomData,
                            });
                        }

                        // Old projections without byte offsets are rebuilt.
                        // New projections append ONLY the committed source
                        // suffix, validating the chain link to the last
                        // projected hash. Prefix trust is explicit here.
                        if let Some(offset) = summary.source_log_bytes
                            && offset < store.log_bytes()
                        {
                            let suffix = (|| -> Result<Self, String> {
                                let mut input = File::open(store.dir.join("events.jsonl"))
                                    .map_err(|e| e.to_string())?;
                                if offset > 0 {
                                    input
                                        .seek(SeekFrom::Start(offset - 1))
                                        .map_err(|e| e.to_string())?;
                                    let mut delimiter = [0u8];
                                    input
                                        .read_exact(&mut delimiter)
                                        .map_err(|e| e.to_string())?;
                                    if delimiter != [b'\n'] {
                                        return Err("projection offset not aligned".into());
                                    }
                                }
                                input
                                    .seek(SeekFrom::Start(offset))
                                    .map_err(|e| e.to_string())?;
                                let mut input =
                                    BufReader::new(input.take(store.log_bytes() - offset));
                                let mut output = OpenOptions::new()
                                    .append(true)
                                    .open(&path)
                                    .map_err(|e| e.to_string())?;
                                let mut line = Vec::new();
                                let mut cursor = offset;
                                let mut previous = summary.event_hash;
                                let mut sequence = summary.sequence;
                                loop {
                                    let n = read_record_limited(&mut input, &mut line)?;
                                    if n == 0 {
                                        break;
                                    }
                                    if !line.ends_with(b"\n") {
                                        return Err("incomplete projection source suffix".into());
                                    }
                                    cursor += n as u64;
                                    let event: Entry<W, W::Action> =
                                        serde_json::from_slice(&line).map_err(|e| e.to_string())?;
                                    if event.previous_hash != previous
                                        || event.sequence != sequence + 1
                                    {
                                        return Err("projection suffix chain mismatch".into());
                                    }
                                    let mut item = EventSummary::from_entry(&event);
                                    item.source_log_bytes = Some(cursor);
                                    serde_json::to_writer(&mut output, &item)
                                        .map_err(|e| e.to_string())?;
                                    output.write_all(b"\n").map_err(|e| e.to_string())?;
                                    previous = event.hash;
                                    sequence += 1;
                                }
                                if cursor != store.log_bytes()
                                    || sequence + 1 != store.sequence()
                                    || previous != store.chain_head()
                                {
                                    return Err("projection suffix ended at different head".into());
                                }
                                output.flush().map_err(|e| e.to_string())?;
                                Ok(Self {
                                    file: Mutex::new(output),
                                    _world: PhantomData,
                                })
                            })();
                            if let Ok(observer) = suffix {
                                return Ok(observer);
                            }
                        }
                    }
                }
            }
        }
        Self::recover(&store.dir)
    }

    /// Rebuild a derived projection from fully verified authoritative events.
    /// Run only while holding the trial's writer lease.
    pub fn recover(dir: impl AsRef<Path>) -> Result<Self, String> {
        let dir = dir.as_ref();
        let verified = Store::<W, W::Action>::inspect_read_only(dir)?;
        let path = Self::path(dir);
        atomic_write_with(&path, |out| write_summaries::<W>(dir, &verified, out))?;
        let file = OpenOptions::new()
            .append(true)
            .open(path)
            .map_err(|e| e.to_string())?;
        Ok(Self {
            file: Mutex::new(file),
            _world: PhantomData,
        })
    }
    fn write_one(&self, entry: &Entry<W, W::Action>, offset: Option<u64>) -> Result<(), String> {
        let mut output = self
            .file
            .lock()
            .map_err(|_| "observability lock poisoned")?;
        let mut summary = EventSummary::from_entry(entry);
        summary.source_log_bytes = offset;
        serde_json::to_writer(&mut *output, &summary).map_err(|e| e.to_string())?;
        output
            .write_all(b"\n")
            .and_then(|_| output.flush())
            .map_err(|e| e.to_string())
    }
}

impl<W> Observer<W> for JsonlObserver<W>
where
    W: World + Serialize + DeserializeOwned,
    W::Action: Serialize + DeserializeOwned + Clone,
{
    fn committed(&self, entry: &Entry<W, W::Action>) -> Result<(), String> {
        self.write_one(entry, None)
    }
    fn committed_with_log_offset(
        &self,
        entry: &Entry<W, W::Action>,
        end: u64,
    ) -> Result<(), String> {
        self.write_one(entry, Some(end))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        builtins::{Counter, Increment, kind},
        core::AgentId,
        experiment::Trial,
        runtime::ScriptedBackend,
        scheduler::Scheduling,
    };
    use std::{fs, sync::Arc};

    #[tokio::test]
    async fn summaries_redact_actions_and_recover_idempotently() {
        let dir = std::env::temp_dir().join(format!("hiyal-r4-observe-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        {
            let mut trial = Trial::create(
                &dir,
                "observation",
                1,
                &kind(1, false, "fixed-v1", "scripted").unwrap(),
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
            trial.observe_with(Arc::new(JsonlObserver::recover(&dir).unwrap()));
            trial.run_attempts(3).await.unwrap();
            trial.checkpoint().unwrap();
        }
        let first = fs::read(JsonlObserver::<Counter>::path(&dir)).unwrap();
        assert_eq!(first.iter().filter(|b| **b == b'\n').count(), 3);
        let text = String::from_utf8(first.clone()).unwrap();
        assert!(!text.contains("action"));
        assert!(!text.contains("world"));
        assert!(!text.contains("prompt"));
        let _observer = JsonlObserver::<Counter>::recover(&dir).unwrap();
        assert_eq!(
            fs::read(JsonlObserver::<Counter>::path(&dir)).unwrap(),
            first
        );
        drop(_observer);
        fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn rejected_and_control_records_are_projected_live_and_on_recovery() {
        let dir =
            std::env::temp_dir().join(format!("hiyal-r4-all-outcomes-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        {
            let mut trial = Trial::create(
                &dir,
                "all-outcomes",
                44,
                &kind(1, false, "fixed-v1", "scripted").unwrap(),
                Counter { total: u64::MAX },
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
            trial.observe_with(Arc::new(JsonlObserver::recover(&dir).unwrap()));
            trial.pause_actor(AgentId(0)).unwrap();
            trial.resume_actor(AgentId(0)).unwrap();
            assert_eq!(trial.run_attempts(1).await.unwrap(), 0);
            assert!(trial.diagnostics.is_empty());
            trial.checkpoint().unwrap();
        }
        let live = fs::read(JsonlObserver::<Counter>::path(&dir)).unwrap();
        let kinds: Vec<String> = String::from_utf8(live.clone())
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<EventSummary>(line).unwrap().kind)
            .collect();
        assert_eq!(kinds, ["control", "control", "rejected"]);
        let _: JsonlObserver<Counter> = JsonlObserver::recover(&dir).unwrap();
        assert_eq!(
            fs::read(JsonlObserver::<Counter>::path(&dir)).unwrap(),
            live
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn write_summaries_rejects_truncated_verified_prefix() {
        use std::io::Write;
        let dir = std::env::temp_dir().join(format!("hiyal-r4-truncate-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut trial = Trial::create(
            &dir,
            "truncated",
            2,
            &kind(1, false, "fixed-v1", "scripted").unwrap(),
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
        trial.run_attempts(2).await.unwrap();
        let verified = Store::<Counter, u64>::inspect_read_only(&dir).unwrap();
        // Simulate malicious storage truncation after the read-only verification.
        let path = dir.join("events.jsonl");
        let f = fs::OpenOptions::new().write(true).open(path).unwrap();
        f.set_len(verified.log_bytes - 1).unwrap();
        drop(f);
        let mut out = Vec::new();
        assert!(write_summaries(&dir, &verified, &mut out).is_err());
        out.flush().unwrap();
        drop(trial);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn schema_remains_versioned() {
        assert_eq!(crate::storage::SCHEMA, 1);
        assert_eq!(OBSERVATION_SCHEMA, 1);
    }
}
