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
    assert!(Store::<Counter, u64>::inspect_read_only(&dir).is_err());
    assert!(resume(&dir).is_err());
    fs::remove_dir_all(dir).unwrap();
}
