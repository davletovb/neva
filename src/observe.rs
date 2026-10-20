//! Privacy-safe structured event projection. This non-authoritative JSONL
//! stream contains no actions, world snapshots, messages or prompt bodies.
use crate::{
    core::World,
    experiment::Observer,
    storage::{Entry, Outcome, Store, atomic_write_with},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    fs::{File, OpenOptions},
    io::{BufRead, BufReader, Write},
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
        }
    }
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

    /// Rebuild a derived projection from fully verified authoritative events.
    /// Run only while holding the trial's writer lease.
    pub fn recover(dir: impl AsRef<Path>) -> Result<Self, String> {
        let dir = dir.as_ref();
        let _: crate::storage::ReadOnly<W> = Store::<W, W::Action>::inspect_read_only(dir)?;
        let path = Self::path(dir);
        atomic_write_with(&path, |out| {
            let file = File::open(dir.join("events.jsonl")).map_err(|e| e.to_string())?;
            for line in BufReader::new(file).lines() {
                let line = line.map_err(|e| e.to_string())?;
                if line.is_empty() {
                    return Err("empty committed event".into());
                }
                let event: Entry<W, W::Action> =
                    serde_json::from_str(&line).map_err(|e| e.to_string())?;
                let summary = EventSummary::from_entry(&event);
                serde_json::to_writer(&mut *out, &summary).map_err(|e| e.to_string())?;
                out.write_all(b"\n").map_err(|e| e.to_string())?;
            }
            Ok(())
        })?;
        let file = OpenOptions::new()
            .append(true)
            .open(path)
            .map_err(|e| e.to_string())?;
        Ok(Self {
            file: Mutex::new(file),
            _world: PhantomData,
        })
    }
}

impl<W> Observer<W> for JsonlObserver<W>
where
    W: World + Serialize + DeserializeOwned,
    W::Action: Serialize + DeserializeOwned + Clone,
{
    fn committed(&self, entry: &Entry<W, W::Action>) -> Result<(), String> {
        let mut output = self
            .file
            .lock()
            .map_err(|_| "observability lock poisoned")?;
        serde_json::to_writer(&mut *output, &EventSummary::from_entry(entry))
            .map_err(|e| e.to_string())?;
        output
            .write_all(b"\n")
            .and_then(|_| output.flush())
            .map_err(|e| e.to_string())
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
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn schema_remains_versioned() {
        assert_eq!(crate::storage::SCHEMA, 1);
        assert_eq!(OBSERVATION_SCHEMA, 1);
    }
}
