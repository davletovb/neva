//! Typed, sequential world transitions. World state is staged before commit.
//! A write failure while exporting JSONL never changes an already committed step.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::io::{self, Write};

/// Stable actor identity assigned by the application or registration layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AgentId(pub u64);

/// A stable trial identifier. No UUID or wall clock participates in derivation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TrialId(pub String);

impl TrialId {
    pub fn new(name: &str, seed: u64) -> Self {
        Self(derive_id("trial", &[name, &seed.to_string()]))
    }
}

/// Versioned, length-prefixed SHA-256 identity derivation, not JSON canonicalization.
/// Changes to this algorithm require an explicit identity profile version.
pub fn derive_id(domain: &str, parts: &[&str]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"hiyal:id:v1");
    for part in std::iter::once(domain).chain(parts.iter().copied()) {
        hasher.update((part.len() as u64).to_be_bytes());
        hasher.update(part.as_bytes());
    }
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(&mut hex, "{byte:02x}").expect("writing into String cannot fail");
    }
    hex
}

/// The entire world, including its pending state, must be clonable for staging.
///
/// R1 will replace full cloning with a validated transition/delta mechanism.
/// Worlds must not perform irreversible I/O within observe/apply.
pub trait World: Clone {
    type Action: Clone + Serialize;
    type Observation;

    fn observe(&self, actor: AgentId) -> Result<Self::Observation, String>;
    fn apply(&mut self, actor: AgentId, action: Self::Action) -> Result<(), String>;
}

/// Pure synchronous policy for the first scripted/rule-based slice.
/// Typed asynchronous model/tool requests are defined in effects.rs; execution
/// and effect-to-proposal resolution remain R1 tasks.
pub trait Policy<W: World> {
    fn propose(&self, actor: AgentId, observation: &W::Observation) -> Result<W::Action, String>;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CommittedEvent<A> {
    pub schema_version: u32,
    pub event_id: String,
    pub trial_id: String,
    pub logical_step: u64,
    pub agent_id: AgentId,
    pub action: A,
}

#[derive(Debug, PartialEq, Eq)]
pub enum EngineError {
    DuplicateAgent(AgentId),
    Observe(String),
    Policy(String),
    Transition(String),
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for EngineError {}

/// Scheduling state is cloned during each step so rejected proposals do not
/// consume turns. Pausing an unregistered actor has no effect.
#[derive(Clone, Debug, Default)]
pub struct RoundRobin {
    agents: Vec<AgentId>,
    paused: BTreeSet<AgentId>,
    cursor: usize,
}

impl RoundRobin {
    pub fn new(agents: impl IntoIterator<Item = AgentId>) -> Result<Self, EngineError> {
        let mut schedule = Self::default();
        for agent in agents {
            schedule.add(agent)?;
        }
        Ok(schedule)
    }

    pub fn add(&mut self, actor: AgentId) -> Result<(), EngineError> {
        if self.agents.contains(&actor) {
            return Err(EngineError::DuplicateAgent(actor));
        }
        self.agents.push(actor);
        Ok(())
    }

    pub fn pause(&mut self, actor: AgentId) {
        if self.agents.contains(&actor) {
            self.paused.insert(actor);
        }
    }

    pub fn resume(&mut self, actor: AgentId) {
        self.paused.remove(&actor);
    }

    pub fn next(&mut self) -> Option<AgentId> {
        for _ in 0..self.agents.len() {
            let actor = self.agents[self.cursor];
            self.cursor = (self.cursor + 1) % self.agents.len();
            if !self.paused.contains(&actor) {
                return Some(actor);
            }
        }
        None
    }
}

/// Minimal in-memory event collector. A durable event-sourced log, resume,
/// checkpoint offsets, and crash-atomic writes are NOT provided yet.
pub struct Engine<W: World, P: Policy<W>> {
    pub trial: TrialId,
    pub world: W,
    pub policy: P,
    pub schedule: RoundRobin,
    pub events: Vec<CommittedEvent<W::Action>>,
}

impl<W: World, P: Policy<W>> Engine<W, P> {
    pub fn new(
        trial: TrialId,
        world: W,
        policy: P,
        agents: impl IntoIterator<Item = AgentId>,
    ) -> Result<Self, EngineError> {
        Ok(Self {
            trial,
            world,
            policy,
            schedule: RoundRobin::new(agents)?,
            events: Vec::new(),
        })
    }

    /// select -> observe -> propose -> validate/apply to clone -> commit.
    /// On any failure, world, event history and scheduler are unchanged.
    pub fn step(&mut self) -> Result<Option<CommittedEvent<W::Action>>, EngineError> {
        let mut staged_schedule = self.schedule.clone();
        let Some(actor) = staged_schedule.next() else {
            return Ok(None);
        };
        let observation = self.world.observe(actor).map_err(EngineError::Observe)?;
        let action = self
            .policy
            .propose(actor, &observation)
            .map_err(EngineError::Policy)?;
        let mut staged_world = self.world.clone();
        staged_world
            .apply(actor, action.clone())
            .map_err(EngineError::Transition)?;
        let logical_step = self.events.len() as u64;
        let event = CommittedEvent {
            schema_version: 1,
            event_id: derive_id(
                "event",
                &[&self.trial.0, &logical_step.to_string(), &actor.0.to_string()],
            ),
            trial_id: self.trial.0.clone(),
            logical_step,
            agent_id: actor,
            action,
        };
        self.world = staged_world;
        self.schedule = staged_schedule;
        self.events.push(event.clone());
        Ok(Some(event))
    }

    /// Prototype export only: serde_json output is not RFC 8785 canonical,
    /// and this method does not provide durable append/commit semantics.
    pub fn write_jsonl(&self, mut output: impl Write) -> io::Result<()> {
        for event in &self.events {
            serde_json::to_writer(&mut output, event).map_err(io::Error::other)?;
            output.write_all(b"\n")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Counter {
        value: u64,
        reject: bool,
    }

    impl World for Counter {
        type Action = u64;
        type Observation = u64;

        fn observe(&self, _: AgentId) -> Result<u64, String> {
            Ok(self.value)
        }

        fn apply(&mut self, _: AgentId, increment: u64) -> Result<(), String> {
            if self.reject {
                return Err("rejected".into());
            }
            self.value += increment;
            Ok(())
        }
    }

    struct Increment;
    impl Policy<Counter> for Increment {
        fn propose(&self, _: AgentId, _: &u64) -> Result<u64, String> {
            Ok(1)
        }
    }

    fn engine(reject: bool) -> Engine<Counter, Increment> {
        Engine::new(
            TrialId::new("golden-two-agent", 7),
            Counter { value: 0, reject },
            Increment,
            [AgentId(0), AgentId(1)],
        )
        .unwrap()
    }

    #[test]
    fn scripted_golden_fixture_and_repeatability() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../fixtures/golden/scripted-two-agent-v1.json"))
                .unwrap();
        let mut first = engine(false);
        let mut second = engine(false);
        for _ in 0..4 {
            first.step().unwrap();
            second.step().unwrap();
        }
        let observed: Vec<u64> = first.events.iter().map(|e| e.agent_id.0).collect();
        let expected: Vec<u64> = fixture["expected_actor_order"]
            .as_array().unwrap().iter().map(|n| n.as_u64().unwrap()).collect();
        assert_eq!(observed, expected);
        assert_eq!(first.trial.0, fixture["expected_trial_id"].as_str().unwrap());
        let expected_ids = fixture["expected_event_ids"].as_array().unwrap();
        for (event, expected_id) in first.events.iter().zip(expected_ids) {
            assert_eq!(event.event_id, expected_id.as_str().unwrap());
        }
        assert_eq!(first.world.value, fixture["expected_state"]["value"].as_u64().unwrap());
        assert_eq!(first.events, second.events);
        let mut a = Vec::new();
        let mut b = Vec::new();
        first.write_jsonl(&mut a).unwrap();
        second.write_jsonl(&mut b).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn rejection_is_atomic_including_scheduler_cursor() {
        let mut e = engine(true);
        assert_eq!(e.step().unwrap_err(), EngineError::Transition("rejected".into()));
        assert!(e.events.is_empty());
        assert_eq!(e.world.value, 0);
        e.world.reject = false;
        assert_eq!(e.step().unwrap().unwrap().agent_id, AgentId(0));
    }

    #[test]
    fn duplicate_and_all_paused_have_defined_behavior() {
        assert!(matches!(
            RoundRobin::new([AgentId(3), AgentId(3)]),
            Err(EngineError::DuplicateAgent(AgentId(3)))
        ));
        let mut e = engine(false);
        e.schedule.pause(AgentId(0));
        e.schedule.pause(AgentId(1));
        assert!(e.step().unwrap().is_none());
        assert_eq!(e.world.value, 0);
        assert!(e.events.is_empty());
        e.schedule.resume(AgentId(1));
        assert_eq!(e.step().unwrap().unwrap().agent_id, AgentId(1));
    }

    #[test]
    fn length_prefixes_disambiguate_identity_components() {
        assert_ne!(derive_id("x", &["ab", "c"]), derive_id("x", &["a", "bc"]));
        assert_ne!(TrialId::new("a", 1), TrialId::new("a", 2));
    }
}
