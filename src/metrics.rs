//! Portable, incremental CSV projections of committed simulation events.
//! CSV is derived state, never authoritative input to replay. Rebuild after
//! restart to recover an observer crash without duplicating metric samples.

use crate::{
    core::World,
    experiment::Observer,
    storage::{Entry, Outcome, Store, atomic_write_with, read_record_limited},
};
use serde::{Serialize, de::DeserializeOwned};
use std::{
    fs::{File, OpenOptions},
    io::{BufReader, Read, Write},
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::Mutex,
};

pub const HEADER: &str = "trial_id,sequence,logical_step,agent_id,metric,value\n";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Metric {
    pub name: String,
    /// Exact integer/decimal text, never an unlabelled f64.
    pub value: String,
}

pub trait MetricProjector<W: World>: Send + Sync {
    fn project(&self, world: &W) -> Result<Vec<Metric>, String>;
}

fn escape_cell(value: &str) -> String {
    if value
        .chars()
        .any(|ch| matches!(ch, ',' | '"' | '\n' | '\r'))
    {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}

pub struct CsvMetrics<W: World, M: MetricProjector<W>> {
    file: Mutex<File>,
    projector: M,
    _world: PhantomData<fn() -> W>,
}

impl<W, M> CsvMetrics<W, M>
where
    W: World + Serialize + DeserializeOwned,
    W::Action: Serialize + DeserializeOwned + Clone,
    M: MetricProjector<W>,
{
    fn write_event(
        writer: &mut impl Write,
        event: &Entry<W, W::Action>,
        projector: &M,
    ) -> Result<(), String> {
        let Outcome::Accepted { world, .. } = &event.outcome else {
            return Ok(());
        };
        for metric in projector.project(world)? {
            if metric.name.is_empty() || metric.name.contains('\0') {
                return Err("invalid metric name".into());
            }
            let row = format!(
                "{},{},{},{},{},{}\n",
                escape_cell(&event.trial_id),
                event.sequence,
                event.logical_step,
                event.actor.0,
                escape_cell(&metric.name),
                escape_cell(&metric.value),
            );
            writer
                .write_all(row.as_bytes())
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Verify source history, then rebuild the non-authoritative projection in
    /// constant memory using a streamed event reader; safe after a torn CSV.
    /// Caller holds the parent trial's exclusive writer lease.
    pub fn recover(dir: impl AsRef<Path>, projector: M) -> Result<Self, String> {
        let dir = dir.as_ref();
        let verified = Store::<W, W::Action>::inspect_read_only(dir)?;
        let target = dir.join("metrics.csv");
        atomic_write_with(&target, |output| {
            output
                .write_all(HEADER.as_bytes())
                .map_err(|e| e.to_string())?;
            let input = File::open(dir.join("events.jsonl")).map_err(|e| e.to_string())?;
            let mut reader = BufReader::new(input.take(verified.log_bytes));
            let mut line = Vec::new();
            let mut read_bytes = 0u64;
            loop {
                let count = read_record_limited(&mut reader, &mut line)?;
                if count == 0 {
                    break;
                }
                if !line.ends_with(b"\n") {
                    return Err("incomplete event record during metrics recovery".into());
                }
                read_bytes += count as u64;
                let event: Entry<W, W::Action> =
                    serde_json::from_slice(&line).map_err(|e| e.to_string())?;
                Self::write_event(output, &event, &projector)?;
            }
            if read_bytes != verified.log_bytes {
                return Err("event prefix changed during metrics recovery".into());
            }
            Ok(())
        })?;
        let file = OpenOptions::new()
            .append(true)
            .open(&target)
            .map_err(|e| e.to_string())?;
        Ok(Self {
            file: Mutex::new(file),
            projector,
            _world: PhantomData,
        })
    }

    pub fn csv_path(dir: impl AsRef<Path>) -> PathBuf {
        dir.as_ref().join("metrics.csv")
    }
}
impl<W, M> Observer<W> for CsvMetrics<W, M>
where
    W: World + Serialize + DeserializeOwned,
    W::Action: Serialize + DeserializeOwned + Clone,
    M: MetricProjector<W>,
{
    fn committed(&self, event: &Entry<W, W::Action>) -> Result<(), String> {
        let mut file = self.file.lock().map_err(|_| "metrics lock poisoned")?;
        Self::write_event(&mut *file, event, &self.projector)?;
        // Derived data is reconstructed from fsynced events on restart.
        // No second synchronous disk barrier on every accepted action.
        file.flush().map_err(|e| e.to_string())
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
    use std::fs;
    use std::sync::Arc;

    struct CounterProjection;
    impl MetricProjector<Counter> for CounterProjection {
        fn project(&self, world: &Counter) -> Result<Vec<Metric>, String> {
            Ok(vec![Metric {
                name: "total".into(),
                value: world.total.to_string(),
            }])
        }
    }

    #[tokio::test]
    async fn csv_is_incremental_exact_and_restart_idempotent() {
        let dir = std::env::temp_dir().join(format!("hiyal-r2-metrics-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        {
            let mut trial = Trial::create(
                &dir,
                "metrics",
                11,
                &kind(1, false, "fixed-v1", "scripted").unwrap(),
                Counter { total: 0 },
                Increment {
                    amount: 1,
                    model: false,
                    model_id: "fixed-v1".into(),
                    provider: "scripted".into(),
                },
                Arc::new(ScriptedBackend),
                vec![AgentId(0), AgentId(1)],
                Scheduling::RoundRobin,
            )
            .unwrap();
            let sink = Arc::new(CsvMetrics::recover(&dir, CounterProjection).unwrap());
            trial.observe_with(sink);
            trial.run_attempts(5).await.unwrap();
            trial.checkpoint().unwrap();
        }
        let expected = fs::read(CsvMetrics::<Counter, CounterProjection>::csv_path(&dir)).unwrap();
        assert_eq!(std::str::from_utf8(&expected).unwrap().lines().count(), 6);
        {
            let _trial = Trial::resume(
                &dir,
                &kind(1, false, "fixed-v1", "scripted").unwrap(),
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
            let _recovered =
                CsvMetrics::<Counter, CounterProjection>::recover(&dir, CounterProjection).unwrap();
        }
        assert_eq!(fs::read(dir.join("metrics.csv")).unwrap(), expected);
        fs::remove_dir_all(dir).unwrap();
    }
}
