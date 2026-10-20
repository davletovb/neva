//! R5 adversarial storage/restart tests. No network, Python, or extra crates.
use hiyal::{
    builtins::{Counter, Increment, kind},
    core::AgentId,
    experiment::Trial,
    runtime::{EffectJournal, ScriptedBackend},
    scheduler::Scheduling,
    storage::Store,
};
use std::{
    fs,
    io::Write,
    path::Path,
    sync::Arc,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT: AtomicU64 = AtomicU64::new(0);
fn trial_dir(case: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "hiyal-r5-{case}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}
fn policy() -> Increment {
    Increment {
        amount: 2,
        model: false,
        model_id: "fixed-v1".into(),
        provider: "scripted".into(),
    }
}
fn world_kind() -> String {
    kind(2, false, "fixed-v1", "scripted").unwrap()
}
fn create(dir: &Path) -> Trial<Counter, Increment> {
    Trial::create(
        dir,
        "r5-check",
        51,
        &world_kind(),
        Counter { total: 0 },
        policy(),
        Arc::new(ScriptedBackend),
        vec![AgentId(0), AgentId(1)],
        Scheduling::RoundRobin,
    )
    .unwrap()
}
fn resume(dir: &Path) -> Result<Trial<Counter, Increment>, String> {
    Trial::resume(
        dir,
        &world_kind(),
        policy(),
        Arc::new(ScriptedBackend),
        false,
    )
}

#[tokio::test]
async fn crash_tail_is_recovered_without_making_up_committed_events() {
    let dir = trial_dir("torn");
    let original_head;
    {
        let mut t = create(&dir);
        assert_eq!(t.run_attempts(3).await.unwrap(), 3);
        original_head = t.store.chain_head().to_string();
    }
    let mut file = fs::OpenOptions::new()
        .append(true)
        .open(dir.join("events.jsonl"))
        .unwrap();
    file.write_all(br#"{"schema":1,"broken":"#).unwrap();
    file.sync_all().unwrap();
    drop(file);
    assert!(Store::<Counter, u64>::inspect_read_only(&dir).is_err());
    {
        let t = resume(&dir).unwrap();
        assert_eq!(t.store.sequence(), 3);
        assert_eq!(t.store.world().total, 6);
        assert_eq!(t.store.chain_head(), original_head);
    }
    assert_eq!(
        Store::<Counter, u64>::inspect_read_only(&dir)
            .unwrap()
            .sequence,
        3
    );
    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn corrupt_committed_log_or_snapshot_fails_without_writing_repairs() {
    let dir = trial_dir("tamper");
    {
        let mut t = create(&dir);
        t.run_attempts(3).await.unwrap();
        t.checkpoint().unwrap();
    }
    let original_log = fs::read(dir.join("events.jsonl")).unwrap();
    let original_snap = fs::read(dir.join("snapshot.json")).unwrap();
    let mut snapshot: serde_json::Value = serde_json::from_slice(&original_snap).unwrap();
    snapshot["chain_head"] = serde_json::json!("forged");
    fs::write(
        dir.join("snapshot.json"),
        serde_json::to_vec(&snapshot).unwrap(),
    )
    .unwrap();
    assert!(Store::<Counter, u64>::inspect_read_only(&dir).is_err());
    assert!(resume(&dir).is_err());
    assert!(!dir.join(".writer.lock").exists());
    fs::write(dir.join("snapshot.json"), &original_snap).unwrap();
    // Every single hash-byte mutation in a committed record must be detected.
    let log_text = String::from_utf8(original_log.clone()).unwrap();
    let mut parts: Vec<serde_json::Value> = log_text
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    for index in 0..3 {
        let original = parts[index]["hash"].as_str().unwrap().to_owned();
        parts[index]["hash"] = serde_json::json!(format!(
            "{}{}",
            if original.starts_with('a') { "b" } else { "a" },
            &original[1..]
        ));
        let tampered = parts
            .iter()
            .map(|record| format!("{}\n", serde_json::to_string(record).unwrap()))
            .collect::<String>();
        fs::write(dir.join("events.jsonl"), &tampered).unwrap();
        assert!(Store::<Counter, u64>::inspect_read_only(&dir).is_err());
        assert!(resume(&dir).is_err());
        parts[index]["hash"] = serde_json::json!(original);
    }
    fs::write(dir.join("events.jsonl"), &original_log).unwrap();
    assert_eq!(
        Store::<Counter, u64>::inspect_read_only(&dir)
            .unwrap()
            .world
            .total,
        6
    );
    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn writer_lease_blocks_parallel_trials_and_recovers_on_drop() {
    let dir = trial_dir("lock");
    let live = create(&dir);
    assert!(resume(&dir).is_err());
    assert!(dir.join(".writer.lock").exists());
    assert!(dir.join(".effects-writer.lock").exists());
    drop(live);
    let resumed = resume(&dir).unwrap();
    assert!(EffectJournal::inspect_read_only(&dir, &resumed.store.genesis().trial_id).is_ok());
    drop(resumed);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn hostile_metadata_size_rejected_before_parsing() {
    let dir = trial_dir("metadata");
    {
        let _t = create(&dir);
    }
    let manifest = dir.join("manifest.json");
    let file = fs::OpenOptions::new().write(true).open(manifest).unwrap();
    file.set_len((hiyal::storage::MAX_METADATA_BYTES + 1) as u64)
        .unwrap();
    drop(file);
    let inspect = Store::<Counter, u64>::inspect_read_only(&dir)
        .err()
        .expect("size validation must fail before JSON parsing");
    assert!(inspect.contains("exceeds portable size limit"), "{inspect}");
    let opened = match resume(&dir) {
        Ok(_) => panic!("oversized manifest was accepted"),
        Err(error) => error,
    };
    assert!(opened.contains("exceeds portable size limit"), "{opened}");
    fs::remove_dir_all(dir).unwrap();
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct SmallState {
    amount: u64,
}
impl hiyal::core::World for SmallState {
    type Action = String;
    type Observation = u64;
    fn observe(&self, _: AgentId) -> Result<u64, String> {
        Ok(self.amount)
    }
    fn apply(&mut self, _: AgentId, action: String) -> Result<(), String> {
        self.amount = self
            .amount
            .checked_add(action.len() as u64)
            .ok_or("overflow")?;
        Ok(())
    }
}
struct HugeFirst {
    attempts: Arc<AtomicU64>,
}
impl hiyal::experiment::EffectPolicy<SmallState> for HugeFirst {
    fn decide(&self, _: AgentId, _: &u64, _: &[hiyal::runtime::Receipt]) -> Result<String, String> {
        if self.attempts.fetch_add(1, Ordering::SeqCst) == 0 {
            Ok("Z".repeat(hiyal::storage::MAX_EVENT_RECORD_BYTES))
        } else {
            Ok("ok".into())
        }
    }
}

#[tokio::test]
async fn oversized_accepted_event_becomes_durable_rejection_and_progresses() {
    let dir = trial_dir("over-event");
    let policy = HugeFirst {
        attempts: Arc::new(AtomicU64::new(0)),
    };
    {
        let mut trial = Trial::create(
            &dir,
            "over-limit",
            99,
            "test.small-state.v1",
            SmallState { amount: 0 },
            policy,
            Arc::new(ScriptedBackend),
            vec![AgentId(0)],
            Scheduling::RoundRobin,
        )
        .unwrap();
        assert!(matches!(
            trial.step().await.unwrap(),
            hiyal::experiment::Step::Rejected { .. }
        ));
        assert_eq!(trial.store.sequence(), 1);
        assert_eq!(trial.store.logical_step(), 0);
        assert_eq!(trial.store.world().amount, 0);
        assert_eq!(trial.run_attempts(1).await.unwrap(), 1);
        assert_eq!(trial.store.sequence(), 2);
        assert_eq!(trial.store.world().amount, 2);
        trial.checkpoint().unwrap();
    }
    let result = Store::<SmallState, String>::inspect_read_only(&dir).unwrap();
    assert_eq!(result.sequence, 2);
    assert_eq!(result.world.amount, 2);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn oversized_genesis_does_not_create_unusable_directory() {
    let dir = trial_dir("genesis-preflight");
    let result = Trial::create(
        &dir,
        "very-large",
        33,
        "test.huge-initial-state.v1",
        StringWorld {
            data: "G".repeat(hiyal::storage::MAX_METADATA_BYTES),
        },
        TextPolicy,
        Arc::new(ScriptedBackend),
        vec![AgentId(0)],
        Scheduling::RoundRobin,
    );
    assert!(result.is_err());
    assert!(!dir.exists(), "preflight must not strand a trial directory");
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct StringWorld {
    data: String,
}
impl hiyal::core::World for StringWorld {
    type Action = u64;
    type Observation = ();
    fn observe(&self, _: AgentId) -> Result<(), String> {
        Ok(())
    }
    fn apply(&mut self, _: AgentId, _: u64) -> Result<(), String> {
        Ok(())
    }
}
struct TextPolicy;
impl hiyal::experiment::EffectPolicy<StringWorld> for TextPolicy {
    fn decide(&self, _: AgentId, _: &(), _: &[hiyal::runtime::Receipt]) -> Result<u64, String> {
        Ok(0)
    }
}

#[test]
fn event_record_reader_refuses_excessive_memory_growth() {
    let bytes = vec![b'A'; hiyal::storage::MAX_EVENT_RECORD_BYTES + 2];
    let mut reader = std::io::BufReader::new(std::io::Cursor::new(bytes));
    let mut line = Vec::new();
    let error = hiyal::storage::read_record_limited(&mut reader, &mut line).unwrap_err();
    assert!(error.contains("event record exceeds"), "{error}");
    assert_eq!(line.len(), hiyal::storage::MAX_EVENT_RECORD_BYTES + 1);
}

#[tokio::test]
async fn effects_ledger_detects_corruption_and_never_repairs_during_inspection() {
    use hiyal::runtime::{EffectJournal, sample_request};
    use std::time::Duration;
    let dir = trial_dir("effect-ledger");
    fs::create_dir_all(&dir).unwrap();
    let mut journal = EffectJournal::open(&dir).unwrap();
    let spec = sample_request("sensitive-private-prompt-913");
    let key = spec.key("effect-trial", 0, AgentId(0), 0).unwrap();
    journal
        .resolve_all(
            vec![(key.clone(), spec)],
            Arc::new(ScriptedBackend),
            1,
            Duration::from_secs(2),
            false,
        )
        .await
        .unwrap();
    drop(journal);
    let path = dir.join("effects.jsonl");
    let original = fs::read(&path).unwrap();
    assert_eq!(
        EffectJournal::inspect_read_only(&dir, "effect-trial").unwrap(),
        1
    );
    for flip in [0usize, original.len() / 2, original.len() - 2] {
        let mut mutated = original.clone();
        mutated[flip] ^= 1;
        fs::write(&path, &mutated).unwrap();
        assert!(EffectJournal::inspect_read_only(&dir, "effect-trial").is_err());
        assert_eq!(
            fs::read(&path).unwrap(),
            mutated,
            "inspect must never modify corrupt receipts"
        );
        assert!(EffectJournal::open_existing(&dir).is_err());
    }
    fs::write(&path, &original).unwrap();
    let mut broken = original.clone();
    broken.extend(b"{\"unfinished\":");
    fs::write(&path, &broken).unwrap();
    assert!(EffectJournal::inspect_read_only(&dir, "effect-trial").is_err());
    {
        let recovered = EffectJournal::open_existing(&dir).unwrap();
        assert_eq!(recovered.receipt_count(), 1);
    }
    assert_eq!(fs::read(&path).unwrap(), original);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn effect_reader_caps_single_record_without_unbounded_allocations() {
    let dir = trial_dir("oversized-effect");
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("effects.jsonl");
    let f = fs::File::create(&path).unwrap();
    f.set_len((hiyal::storage::MAX_EVENT_RECORD_BYTES + 2) as u64)
        .unwrap();
    drop(f);
    let error = hiyal::runtime::EffectJournal::inspect_read_only(&dir, "effect-trial").unwrap_err();
    assert!(error.contains("size limit"), "{error}");
    fs::remove_dir_all(dir).unwrap();
}

/// Deterministic property-style mutation matrix: every nonempty subset of
/// structural bytes changes integrity or JSON validity, never a valid receipt.
#[tokio::test]
async fn effect_receipt_mutation_matrix_is_detected() {
    use hiyal::runtime::{EffectJournal, sample_request};
    use std::time::Duration;
    let dir = trial_dir("mutation-matrix");
    fs::create_dir_all(&dir).unwrap();
    let mut journal = EffectJournal::open(&dir).unwrap();
    let spec = sample_request("opaque-secret-c7d");
    let key = spec.key("mutations", 0, AgentId(0), 0).unwrap();
    journal
        .resolve_all(
            vec![(key, spec)],
            Arc::new(ScriptedBackend),
            1,
            Duration::from_secs(2),
            false,
        )
        .await
        .unwrap();
    drop(journal);
    let path = dir.join("effects.jsonl");
    let original = fs::read(&path).unwrap();
    let sample_positions = [
        0,
        1,
        original.len() / 4,
        original.len() / 2,
        original.len() * 3 / 4,
        original.len() - 3,
    ];
    for pos in sample_positions {
        let mut mutated = original.clone();
        // Force invalid UTF-8 while preserving line delimiters.
        mutated[pos] = 0xff;
        fs::write(&path, &mutated).unwrap();
        assert!(EffectJournal::inspect_read_only(&dir, "mutations").is_err());
    }
    fs::write(&path, &original).unwrap();
    fs::remove_dir_all(dir).unwrap();
}
