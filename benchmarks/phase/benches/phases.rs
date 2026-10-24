//! Criterion timings of Hiyal engine phases on three synthetic world shapes.
//! Warm runs only: many iterations in one process. Cold, fresh-process
//! replay and allocation counts come from the `phase-report` binary.
//!
//! cargo bench --locked                      # everything
//! cargo bench --locked -- 'replay/'         # one group (regex filter)
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use hiyal::storage::{Durability, Store};
use hiyal_phase_bench::{
    ConstantWorld, GrowingWorld, KeyedWorld, Scratch, Workload, build_journal, new_trial, runtime,
    stage_event,
};
use serde::{Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use std::time::Duration;

/// A small and a large world of each shape.
fn sizes(shape: &str) -> [usize; 2] {
    match shape {
        "constant" => [64, 4096],
        "growing" => [100, 2000],
        _ => [16, 512],
    }
}

fn computation<W: Workload>(c: &mut Criterion, shape: &str)
where
    W::Action: Serialize + DeserializeOwned + Clone + 'static,
{
    for size in sizes(shape) {
        let world = W::build(size);
        let bytes = hiyal::storage::stable_bytes(&world).expect("serialize");
        let id = format!("{shape}/{size}");

        let mut group = c.benchmark_group("serialize");
        group.throughput(Throughput::Bytes(bytes.len() as u64));
        group.bench_function(BenchmarkId::from_parameter(&id), |b| {
            b.iter(|| hiyal::storage::stable_bytes(&world).expect("serialize"))
        });
        group.finish();

        let mut group = c.benchmark_group("digest");
        group.throughput(Throughput::Bytes(bytes.len() as u64));
        group.bench_function(BenchmarkId::from_parameter(&id), |b| {
            b.iter(|| hiyal::storage::digest(&world).expect("digest"))
        });
        group.finish();

        let mut group = c.benchmark_group("sha256");
        group.throughput(Throughput::Bytes(bytes.len() as u64));
        group.bench_function(BenchmarkId::from_parameter(&id), |b| {
            b.iter(|| Sha256::digest(&bytes))
        });
        group.finish();

        let mut group = c.benchmark_group("apply");
        group.throughput(Throughput::Elements(1));
        group.bench_function(BenchmarkId::from_parameter(&id), |b| {
            // The clone is made, and the result dropped, outside the timer.
            // One clone per iteration, as in phase-report: a batch of clones
            // changes the heap layout, and with it whether a growing world's
            // push can extend its buffer in place. Per-iteration timing adds
            // the timer's own tens of nanoseconds; phase-report subtracts it.
            b.iter_batched(
                || world.clone(),
                |mut next| {
                    let step = next.step();
                    let actor = step % hiyal_phase_bench::AGENTS;
                    next.apply(hiyal::core::AgentId(actor), W::action_for(step, actor))
                        .expect("apply");
                    next
                },
                BatchSize::PerIteration,
            )
        });
        group.finish();

        let mut group = c.benchmark_group("stage");
        group.throughput(Throughput::Elements(1));
        group.bench_function(BenchmarkId::from_parameter(&id), |b| {
            b.iter(|| stage_event(&world).expect("stage"))
        });
        group.finish();

        #[cfg(feature = "undo")]
        {
            let mut group = c.benchmark_group("stage_in_place");
            group.throughput(Throughput::Elements(1));
            let mut staged = world.clone();
            group.bench_function(BenchmarkId::from_parameter(&id), |b| {
                b.iter(|| hiyal_phase_bench::stage_event_in_place(&mut staged).expect("stage"))
            });
            group.finish();
        }
    }
}

fn storage<W: Workload>(c: &mut Criterion, shape: &str)
where
    W::Action: Serialize + DeserializeOwned + Clone + 'static,
{
    let rt = runtime();
    for size in sizes(shape) {
        let id = format!("{shape}/{size}");
        for (name, durability, samples) in [
            ("commit_buffered", Durability::AtCheckpoint, 30),
            ("commit_fsync", Durability::EveryEvent, 10),
        ] {
            let mut group = c.benchmark_group(name);
            group.sample_size(samples);
            group.throughput(Throughput::Elements(1));
            group.bench_function(BenchmarkId::from_parameter(&id), |b| {
                b.iter_batched(
                    || {
                        let scratch = Scratch::new(name);
                        let trial =
                            new_trial::<W>(scratch.path(), size, durability).expect("trial");
                        (scratch, trial)
                    },
                    |(scratch, mut trial)| {
                        rt.block_on(trial.run_attempts(1)).expect("commit");
                        // Dropped outside the timed routine.
                        (scratch, trial)
                    },
                    BatchSize::PerIteration,
                )
            });
            group.finish();
        }
    }

    let size = sizes(shape)[0];
    for events in [10u64, 100, 500] {
        let journal = Scratch::new("bench-journal");
        build_journal::<W>(journal.path(), size, events).expect("journal");
        let id = format!("{shape}/{size}/{events}");

        let mut group = c.benchmark_group("replay_full");
        group.throughput(Throughput::Elements(events));
        group.bench_function(BenchmarkId::from_parameter(&id), |b| {
            b.iter(|| Store::<W, W::Action>::inspect_read_only(journal.path()).expect("replay"))
        });
        group.finish();

        #[cfg(feature = "replay-cache")]
        {
            let mut group = c.benchmark_group("inspect_cached_hit");
            hiyal::storage::clear_replay_cache();
            Store::<W, W::Action>::inspect_read_only_cached(journal.path()).expect("warm memo");
            group.bench_function(BenchmarkId::from_parameter(&id), |b| {
                b.iter(|| {
                    Store::<W, W::Action>::inspect_read_only_cached(journal.path()).expect("hit")
                })
            });
            group.finish();
        }

        // Skips verification of events before the snapshot: not comparable
        // with replay_full.
        let mut group = c.benchmark_group("open_trusted_snapshot");
        group.bench_function(BenchmarkId::from_parameter(&id), |b| {
            b.iter(|| Store::<W, W::Action>::open_trusted(journal.path()).expect("open"))
        });
        group.finish();
    }
}

fn phases(c: &mut Criterion) {
    computation::<ConstantWorld>(c, "constant");
    computation::<GrowingWorld>(c, "growing");
    computation::<KeyedWorld>(c, "keyed");
    storage::<ConstantWorld>(c, "constant");
    storage::<GrowingWorld>(c, "growing");
    storage::<KeyedWorld>(c, "keyed");
}

fn config() -> Criterion {
    Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
}

criterion_group! {
    name = benches;
    config = config();
    targets = phases
}
criterion_main!(benches);
