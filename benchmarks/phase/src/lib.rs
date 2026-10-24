//! Per-phase Hiyal engine benchmarks: synthetic world shapes, workload
//! helpers, provenance and allocation accounting shared by the Criterion
//! suite (`benches/phases.rs`) and the `phase-report` binary.
//!
//! Only Hiyal's public API is used, and, apart from the `replay-cache`
//! feature, only API that also exists at older revisions, so `compare.sh` can
//! build the same workloads against a baseline commit.
#[cfg(feature = "undo")]
use hiyal::{core::Undo, rng::Streams};
use hiyal::{
    core::{AgentId, Policy, World},
    experiment::{Rules, Trial},
    runtime::ScriptedBackend,
    scheduler::Scheduling,
    storage::Durability,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    time::Instant,
};

pub const AGENTS: u64 = 4;

/// A synthetic world whose next action depends only on the step and actor,
/// so every revision and shape commits the same deterministic history.
pub trait Workload: World<Observation = u64> + Serialize + DeserializeOwned + 'static
where
    Self::Action: Serialize + DeserializeOwned + Clone + 'static,
{
    /// World kind recorded in the genesis.
    const NAME: &'static str;
    /// A world of the given size parameter (see each shape).
    fn build(size: usize) -> Self;
    fn action_for(step: u64, actor: u64) -> Self::Action;
    fn step(&self) -> u64;
}

/// Constant size: `size` slots of state that are updated in place.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConstantWorld {
    pub step: u64,
    pub values: Vec<f64>,
    pub counts: Vec<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bump {
    pub slot: u64,
    pub signal: f64,
}

impl World for ConstantWorld {
    type Action = Bump;
    type Observation = u64;
    fn observe(&self, _: AgentId) -> Result<u64, String> {
        Ok(self.step)
    }
    fn apply(&mut self, _: AgentId, bump: Bump) -> Result<(), String> {
        let slot = usize::try_from(bump.slot).map_err(|e| e.to_string())? % self.values.len();
        self.values[slot] = 0.9 * self.values[slot] + 0.1 * bump.signal;
        self.counts[slot] += 1;
        self.step += 1;
        Ok(())
    }
    #[cfg(feature = "undo")]
    fn apply_undoable(
        &mut self,
        actor: AgentId,
        bump: Bump,
        _: &mut Streams,
    ) -> Result<Undo<Self>, String> {
        let slot = usize::try_from(bump.slot).map_err(|e| e.to_string())? % self.values.len();
        let (value, count, step) = (self.values[slot], self.counts[slot], self.step);
        self.apply(actor, bump)?;
        Ok(Undo::revert(move |world: &mut Self| {
            world.values[slot] = value;
            world.counts[slot] = count;
            world.step = step;
        }))
    }
}

impl Workload for ConstantWorld {
    const NAME: &'static str = "phase-bench.constant.v1";
    fn build(size: usize) -> Self {
        let size = size.max(1);
        Self {
            step: 0,
            values: (0..size)
                .map(|i| 0.5 + i as f64 / (4 * size) as f64)
                .collect(),
            counts: vec![0; size],
        }
    }
    fn action_for(step: u64, actor: u64) -> Bump {
        Bump {
            slot: step.wrapping_mul(31).wrapping_add(actor),
            signal: ((step % 97) as f64) / 97.0,
        }
    }
    fn step(&self) -> u64 {
        self.step
    }
}

/// Growing: one history record appended per event, on top of `size`
/// records at genesis (like a world that stores its own trace).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GrowingWorld {
    pub step: u64,
    pub state: Vec<f64>,
    pub history: Vec<Record>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Record {
    pub step: u64,
    pub actor: u64,
    pub value: f64,
    pub note: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Observe {
    pub value: f64,
}

impl World for GrowingWorld {
    type Action = Observe;
    type Observation = u64;
    fn observe(&self, _: AgentId) -> Result<u64, String> {
        Ok(self.step)
    }
    fn apply(&mut self, actor: AgentId, observe: Observe) -> Result<(), String> {
        let slot = (self.step % self.state.len() as u64) as usize;
        self.state[slot] = 0.9 * self.state[slot] + 0.1 * observe.value;
        self.step += 1;
        self.history.push(Record {
            step: self.step,
            actor: actor.0,
            value: observe.value,
            note: format!("event-{}", self.step),
        });
        Ok(())
    }
    #[cfg(feature = "undo")]
    fn apply_undoable(
        &mut self,
        actor: AgentId,
        observe: Observe,
        _: &mut Streams,
    ) -> Result<Undo<Self>, String> {
        let slot = (self.step % self.state.len() as u64) as usize;
        let (value, step) = (self.state[slot], self.step);
        self.apply(actor, observe)?;
        Ok(Undo::revert(move |world: &mut Self| {
            world.history.pop();
            world.state[slot] = value;
            world.step = step;
        }))
    }
}

impl Workload for GrowingWorld {
    const NAME: &'static str = "phase-bench.growing.v1";
    fn build(size: usize) -> Self {
        Self {
            step: 0,
            state: vec![0.5; 16],
            history: (0..size as u64)
                .map(|i| Record {
                    step: 0,
                    actor: i % AGENTS,
                    value: (i % 89) as f64 / 89.0,
                    note: format!("genesis-{i}"),
                })
                .collect(),
        }
    }
    fn action_for(step: u64, _: u64) -> Observe {
        Observe {
            value: ((step * 7) % 101) as f64 / 101.0,
        }
    }
    fn step(&self) -> u64 {
        self.step
    }
}

/// Many small maps: `size` tables of 8 string-keyed counters, the shape of
/// per-agent or per-stream bookkeeping such as Hiyal's own RNG positions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KeyedWorld {
    pub step: u64,
    pub tables: BTreeMap<String, BTreeMap<String, u64>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Touch {
    pub table: u64,
    pub key: u64,
}

impl World for KeyedWorld {
    type Action = Touch;
    type Observation = u64;
    fn observe(&self, _: AgentId) -> Result<u64, String> {
        Ok(self.step)
    }
    fn apply(&mut self, _: AgentId, touch: Touch) -> Result<(), String> {
        let (table, key) = self.cell(&touch);
        *self.counter(&table, &key)? += 1;
        self.step += 1;
        Ok(())
    }
    #[cfg(feature = "undo")]
    fn apply_undoable(
        &mut self,
        _: AgentId,
        touch: Touch,
        _: &mut Streams,
    ) -> Result<Undo<Self>, String> {
        let (table, key) = self.cell(&touch);
        let counter = self.counter(&table, &key)?;
        let value = *counter;
        *counter += 1;
        let step = self.step;
        self.step += 1;
        Ok(Undo::revert(move |world: &mut Self| {
            *world.counter(&table, &key).expect("touched counter") = value;
            world.step = step;
        }))
    }
}

impl KeyedWorld {
    fn cell(&self, touch: &Touch) -> (String, String) {
        (
            format!("table/{:05}", touch.table % self.tables.len() as u64),
            format!("key/{}", touch.key % 8),
        )
    }
    fn counter(&mut self, table: &str, key: &str) -> Result<&mut u64, String> {
        Ok(self
            .tables
            .get_mut(table)
            .ok_or("missing table")?
            .get_mut(key)
            .ok_or("missing key")?)
    }
}

impl Workload for KeyedWorld {
    const NAME: &'static str = "phase-bench.keyed.v1";
    fn build(size: usize) -> Self {
        Self {
            step: 0,
            tables: (0..size.max(1))
                .map(|t| {
                    (
                        format!("table/{t:05}"),
                        (0..8).map(|k| (format!("key/{k}"), 0)).collect(),
                    )
                })
                .collect(),
        }
    }
    fn action_for(step: u64, actor: u64) -> Touch {
        Touch {
            table: step.wrapping_mul(13).wrapping_add(actor),
            key: step,
        }
    }
    fn step(&self) -> u64 {
        self.step
    }
}

/// Policy proposing `W::action_for(step, actor)`.
pub struct Driver<W>(std::marker::PhantomData<fn() -> W>);
impl<W> Default for Driver<W> {
    fn default() -> Self {
        Self(std::marker::PhantomData)
    }
}
impl<W: Workload> Policy<W> for Driver<W>
where
    W::Action: Serialize + DeserializeOwned + Clone + 'static,
{
    fn propose(&self, actor: AgentId, step: &u64) -> Result<W::Action, String> {
        Ok(W::action_for(*step, actor.0))
    }
}

pub type BenchTrial<W> = Trial<W, Rules<Driver<W>>>;

/// A single-threaded runtime for Hiyal's async trial API.
pub fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("tokio runtime")
}

/// A fresh trial over `W::build(size)` with the given journal durability.
pub fn new_trial<W: Workload>(
    dir: &Path,
    size: usize,
    durability: Durability,
) -> Result<BenchTrial<W>, String>
where
    W::Action: Serialize + DeserializeOwned + Clone + 'static,
{
    let mut trial = Trial::create(
        dir,
        "phase-bench",
        1,
        W::NAME,
        W::build(size),
        Rules(Driver::<W>::default()),
        Arc::new(ScriptedBackend),
        (0..AGENTS).map(AgentId).collect(),
        Scheduling::RoundRobin,
    )?;
    trial.store.set_durability(durability)?;
    Ok(trial)
}

/// A finished trial directory with `events` committed events and a final
/// snapshot, written without per-event fsync.
pub fn build_journal<W: Workload>(dir: &Path, size: usize, events: u64) -> Result<(), String>
where
    W::Action: Serialize + DeserializeOwned + Clone + 'static,
{
    let rt = runtime();
    let mut trial = new_trial::<W>(dir, size, Durability::AtCheckpoint)?;
    let committed = rt.block_on(trial.run_attempts(events))?;
    if committed != events {
        return Err(format!("only {committed} of {events} events committed"));
    }
    trial.checkpoint()
}

/// Applies `count` actions to a clone of `world`: the reducer cost alone.
pub fn apply_actions<W: Workload>(world: &W, count: u64) -> Result<W, String>
where
    W::Action: Serialize + DeserializeOwned + Clone + 'static,
{
    let mut next = world.clone();
    for _ in 0..count {
        let step = next.step();
        let actor = step % AGENTS;
        next.apply(AgentId(actor), W::action_for(step, actor))?;
    }
    Ok(next)
}

/// The work Hiyal does to stage one accepted event, reproduced from public
/// API: clone the world, apply the action, hash the result.
pub fn stage_event<W: Workload>(world: &W) -> Result<(W, String), String>
where
    W::Action: Serialize + DeserializeOwned + Clone + 'static,
{
    let mut next = world.clone();
    let step = next.step();
    let actor = step % AGENTS;
    next.apply(AgentId(actor), W::action_for(step, actor))?;
    let hash = hiyal::storage::digest(&next)?;
    Ok((next, hash))
}

/// The same staging in place, as Hiyal's runtime does it: apply with
/// `World::apply_undoable` and hash the world, then undo, so `world` is
/// unchanged afterwards and the next call stages the same event.
#[cfg(feature = "undo")]
pub fn stage_event_in_place<W: Workload>(world: &mut W) -> Result<String, String>
where
    W::Action: Serialize + DeserializeOwned + Clone + 'static,
{
    let step = world.step();
    let actor = step % AGENTS;
    let mut streams = Streams::new("stage".into(), 0);
    let undo = world.apply_undoable(AgentId(actor), W::action_for(step, actor), &mut streams)?;
    let hash = hiyal::storage::digest(&*world);
    undo.undo(world);
    hash
}

/// A unique scratch directory removed on drop.
pub struct Scratch(pub PathBuf);
impl Scratch {
    pub fn new(tag: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let base = std::env::var_os("HIYAL_PHASE_BENCH_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        Self(base.join(format!(
            "hiyal-phase-{tag}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )))
    }
    pub fn path(&self) -> &Path {
        &self.0
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Size of all regular files under `dir`, in bytes.
pub fn dir_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| match entry.metadata() {
            Ok(m) if m.is_dir() => dir_bytes(&entry.path()),
            Ok(m) => m.len(),
            Err(_) => 0,
        })
        .sum()
}

/// Wall-clock nanoseconds of one call.
pub fn time_ns<T>(f: impl FnOnce() -> T) -> (u64, T) {
    let start = Instant::now();
    let value = f();
    (start.elapsed().as_nanos() as u64, value)
}

/// Minimum duration of one sample of a fast phase.
pub const MIN_SAMPLE: std::time::Duration = std::time::Duration::from_millis(5);

/// Nanoseconds per call of `f`, averaged over enough calls that the sample
/// lasts at least `MIN_SAMPLE`, so timer resolution and scheduler noise do
/// not dominate microsecond-scale phases. The first call only calibrates.
pub fn per_call_ns<T>(mut f: impl FnMut() -> T) -> f64 {
    let (first, _) = time_ns(&mut f);
    let calls = (MIN_SAMPLE.as_nanos() as u64 / first.max(1)).clamp(1, 1_000_000);
    let start = Instant::now();
    for _ in 0..calls {
        std::hint::black_box(f());
    }
    start.elapsed().as_nanos() as f64 / calls as f64
}

/// Counting global allocator: total allocations and bytes, and the peak of
/// live bytes, since the last `AllocStats::reset`. Declare it with
/// `#[global_allocator]` in a binary to enable the counts.
pub struct CountingAlloc;

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

fn grow(bytes: usize) {
    ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    ALLOCATED_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
    let live = LIVE_BYTES.fetch_add(bytes, Ordering::Relaxed) + bytes;
    PEAK_BYTES.fetch_max(live, Ordering::Relaxed);
}

// SAFETY: every call forwards to `System` with the same layout; the counters
// are plain atomics and never allocate.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            grow(layout.size());
        }
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            grow(layout.size());
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new = unsafe { System.realloc(ptr, layout, new_size) };
        if !new.is_null() {
            LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
            grow(new_size);
        }
        new
    }
}

/// Allocation activity of one measured section.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct AllocStats {
    pub allocations: u64,
    pub allocated_bytes: u64,
    /// Highest live heap during the section, above the live heap at its start.
    pub peak_bytes: u64,
}

impl AllocStats {
    /// Start a section: clears the counters and returns its live baseline.
    pub fn reset() -> usize {
        ALLOCATIONS.store(0, Ordering::Relaxed);
        ALLOCATED_BYTES.store(0, Ordering::Relaxed);
        let live = LIVE_BYTES.load(Ordering::Relaxed);
        PEAK_BYTES.store(live, Ordering::Relaxed);
        live
    }
    pub fn since(baseline: usize) -> Self {
        Self {
            allocations: ALLOCATIONS.load(Ordering::Relaxed),
            allocated_bytes: ALLOCATED_BYTES.load(Ordering::Relaxed),
            peak_bytes: PEAK_BYTES.load(Ordering::Relaxed).saturating_sub(baseline) as u64,
        }
    }
    /// Counts of `f`, and its result.
    pub fn measure<T>(f: impl FnOnce() -> T) -> (Self, T) {
        let baseline = Self::reset();
        let value = f();
        (Self::since(baseline), value)
    }
}

pub mod provenance;

#[cfg(test)]
mod tests {
    use super::*;
    use hiyal::storage::Store;

    fn journal_replays<W: Workload>(size: usize)
    where
        W::Action: Serialize + DeserializeOwned + Clone + 'static,
    {
        let scratch = Scratch::new("test");
        build_journal::<W>(scratch.path(), size, 12).unwrap();
        let view = Store::<W, W::Action>::inspect_read_only(scratch.path()).unwrap();
        assert_eq!(view.world.step(), 12);
        let applied = apply_actions(&W::build(size), 12).unwrap();
        assert_eq!(
            hiyal::storage::digest(&view.world).unwrap(),
            hiyal::storage::digest(&applied).unwrap(),
            "{}: the committed history must equal the policy-free one",
            W::NAME
        );
        drop(Store::<W, W::Action>::open_trusted(scratch.path()).unwrap());
    }

    #[test]
    fn every_shape_commits_and_replays_the_same_history() {
        journal_replays::<ConstantWorld>(8);
        journal_replays::<GrowingWorld>(5);
        journal_replays::<KeyedWorld>(4);
    }

    fn len<T: Serialize>(value: &T) -> usize {
        hiyal::storage::stable_bytes(value).unwrap().len()
    }

    #[test]
    fn growing_world_grows_and_the_others_do_not() {
        let constant = ConstantWorld::build(16);
        let growing = GrowingWorld::build(0);
        let keyed = KeyedWorld::build(4);
        let grown = |n| len(&apply_actions(&growing, n).unwrap());
        assert!(grown(100) > 5 * grown(10));
        let constant_len = |n| len(&apply_actions(&constant, n).unwrap());
        assert!(constant_len(100) < 2 * constant_len(10));
        let keyed_len = |n| len(&apply_actions(&keyed, n).unwrap());
        assert!(keyed_len(100) < 2 * keyed_len(10));
    }

    #[cfg(feature = "undo")]
    fn in_place_staging_matches_cloned_staging<W: Workload>(size: usize)
    where
        W::Action: Serialize + DeserializeOwned + Clone + 'static,
    {
        let mut world = apply_actions(&W::build(size), 7).unwrap();
        let before = hiyal::storage::digest(&world).unwrap();
        let (_, cloned) = stage_event(&world).unwrap();
        assert_eq!(stage_event_in_place(&mut world).unwrap(), cloned);
        assert_eq!(
            hiyal::storage::digest(&world).unwrap(),
            before,
            "undo is exact"
        );
    }

    #[cfg(feature = "undo")]
    #[test]
    fn every_shape_undoes_in_place_staging_exactly() {
        in_place_staging_matches_cloned_staging::<ConstantWorld>(64);
        in_place_staging_matches_cloned_staging::<GrowingWorld>(100);
        in_place_staging_matches_cloned_staging::<KeyedWorld>(16);
    }

    #[test]
    fn counting_allocator_type_reports_nothing_without_registration() {
        // Not registered as the global allocator in unit tests: counts stay zero.
        let (stats, v) = AllocStats::measure(|| vec![0u8; 1024]);
        assert_eq!(v.len(), 1024);
        assert_eq!(stats.allocations, 0);
    }
}
