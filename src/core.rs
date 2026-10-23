//! Typed sequential world transitions; successful world actions are staged.
//! A rejected attempt advances scheduling, but cannot partially mutate the world.
//! This R0 prototype is in-memory, NOT a durable event-sourced engine.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::io::{self, Write};

/// JSON-compatible unsigned 64-bit integers must be exact decimal strings.
/// Numeric JSON is deliberately rejected even when the value would fit in f64.
pub mod decimal_u64 {
    use serde::{Deserialize, Deserializer, Serializer, de::Error as _};

    pub fn serialize<S: Serializer>(value: &u64, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&value.to_string())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<u64, D::Error> {
        let text = String::deserialize(deserializer)?;
        let number = text.parse::<u64>().map_err(D::Error::custom)?;
        if text != number.to_string() {
            return Err(D::Error::custom(
                "expected canonical unsigned decimal string",
            ));
        }
        Ok(number)
    }

    pub mod option {
        use super::*;

        pub fn serialize<S: Serializer>(
            value: &Option<u64>,
            serializer: S,
        ) -> Result<S::Ok, S::Error> {
            match value {
                Some(value) => serializer.serialize_some(&value.to_string()),
                None => serializer.serialize_none(),
            }
        }

        pub fn deserialize<'de, D: Deserializer<'de>>(
            deserializer: D,
        ) -> Result<Option<u64>, D::Error> {
            let text = Option::<String>::deserialize(deserializer)?;
            text.map(|text| {
                let value = text.parse::<u64>().map_err(D::Error::custom)?;
                if text != value.to_string() {
                    return Err(D::Error::custom(
                        "expected canonical unsigned decimal string",
                    ));
                }
                Ok(value)
            })
            .transpose()
        }
    }
}

/// Stable actor identity assigned by the application or registration layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AgentId(#[serde(with = "decimal_u64")] pub u64);

/// A stable trial identifier. No UUID or wall clock participates in derivation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TrialId(pub String);

impl TrialId {
    pub fn new(name: &str, seed: u64) -> Self {
        Self(derive_id("trial", &[name, &seed.to_string()]))
    }
}

/// Versioned length-prefixed SHA-256 identities, not JCS canonicalization.
pub fn derive_id(domain: &str, parts: &[&str]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"hiyal:id:v1");
    for part in std::iter::once(domain).chain(parts.iter().copied()) {
        hasher.update((part.len() as u64).to_be_bytes());
        hasher.update(part.as_bytes());
    }
    hex_digest(&hasher.finalize())
}

pub(crate) fn hex_digest(bytes: &[u8]) -> String {
    let mut hex = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut hex, "{byte:02x}").expect("writing into String cannot fail");
    }
    hex
}

/// Explicitly sort object keys *recursively*, independently of the global
/// serde_json `preserve_order` Cargo feature (which downstream crates can
/// enable through feature unification). This is NOT RFC 8785/JCS.
/// Production code writes these bytes directly (`canonical`); this `Value`
/// route is kept as the reference implementation for differential tests.
#[cfg(test)]
pub(crate) fn stable_json_bytes(
    value: &serde_json::Value,
    out: &mut Vec<u8>,
) -> serde_json::Result<()> {
    match value {
        serde_json::Value::Object(object) => {
            out.push(b'{');
            let mut keys: Vec<_> = object.keys().collect();
            keys.sort_unstable();
            for (position, key) in keys.iter().enumerate() {
                if position != 0 {
                    out.push(b',');
                }
                serde_json::to_writer(&mut *out, key)?;
                out.push(b':');
                stable_json_bytes(&object[key.as_str()], out)?;
            }
            out.push(b'}');
        }
        serde_json::Value::Array(values) => {
            out.push(b'[');
            for (position, value) in values.iter().enumerate() {
                if position != 0 {
                    out.push(b',');
                }
                stable_json_bytes(value, out)?;
            }
            out.push(b']');
        }
        scalar => serde_json::to_writer(out, scalar)?,
    }
    Ok(())
}

/// Provisional digest of recursively ordered JSON action data. R1 will use a
/// verified RFC 8785 canonicalizer and reject unsupported numeric shapes.
fn action_digest<A: Serialize>(action: &A) -> Result<String, EngineError> {
    let bytes = crate::canonical::to_stable_bytes(action)
        .map_err(|error| EngineError::Serialization(error.to_string()))?;
    Ok(hex_digest(&Sha256::digest(bytes)))
}

/// The world is cloned to stage action application. Only declared operations
/// may change a world; observe/apply must not produce external side effects.
pub trait World: Clone {
    type Action: Clone + Serialize;
    type Observation;

    fn observe(&self, actor: AgentId) -> Result<Self::Observation, String>;
    fn apply(&mut self, actor: AgentId, action: Self::Action) -> Result<(), String>;
    /// Opt-in replayable world randomness; only named trial streams.
    fn apply_with_streams(
        &mut self,
        actor: AgentId,
        action: Self::Action,
        _streams: &mut crate::rng::Streams,
    ) -> Result<(), String> {
        self.apply(actor, action)
    }
}

/// Pure synchronous policy; async effect resolution remains R1 work.
pub trait Policy<W: World> {
    fn propose(&self, actor: AgentId, observation: &W::Observation) -> Result<W::Action, String>;
    fn propose_with_streams(
        &self,
        actor: AgentId,
        observation: &W::Observation,
        _streams: &mut crate::rng::Streams,
    ) -> Result<W::Action, String> {
        self.propose(actor, observation)
    }
}

/// A successful world transition. The action alone is NOT sufficient to
/// reconstruct arbitrary worlds. An R1 event must include a replayable delta.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CommittedEvent<A> {
    pub schema_version: u32,
    pub event_id: String,
    pub previous_event_id: Option<String>,
    pub action_hash: String,
    pub trial_id: String,
    #[serde(with = "decimal_u64")]
    pub logical_step: u64,
    pub agent_id: AgentId,
    pub action: A,
}

/// An unsuccessful actor attempt, separate from committed world transitions.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RejectedAttempt {
    pub schema_version: u32,
    pub rejection_id: String,
    pub trial_id: String,
    #[serde(with = "decimal_u64")]
    pub attempt_index: u64,
    #[serde(with = "decimal_u64")]
    pub next_logical_step: u64,
    pub agent_id: AgentId,
    pub reason: String,
}

/// Chronological in-memory audit. Rejections are not world state commits.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "record", rename_all = "snake_case")]
pub enum AuditRecord<A> {
    Committed(CommittedEvent<A>),
    Rejected(RejectedAttempt),
}

#[derive(Debug, PartialEq, Eq)]
pub enum EngineError {
    DuplicateAgent(AgentId),
    Observe(String),
    Policy(String),
    Transition(String),
    Serialization(String),
    Exhausted,
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for EngineError {}

/// O(log N) duplicate/paused-membership tests and O(1) cursor movement
/// for an eligible actor. No per-step clone of the entire schedule.
#[derive(Clone, Debug, Default)]
pub struct RoundRobin {
    agents: Vec<AgentId>,
    registered: BTreeSet<AgentId>,
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
        if !self.registered.insert(actor) {
            return Err(EngineError::DuplicateAgent(actor));
        }
        self.agents.push(actor);
        Ok(())
    }

    pub fn pause(&mut self, actor: AgentId) {
        if self.registered.contains(&actor) {
            self.paused.insert(actor);
        }
    }

    pub fn resume(&mut self, actor: AgentId) {
        self.paused.remove(&actor);
    }

    /// Selection consumes scheduling eligibility even on policy/world rejection.
    /// The caller must audit that rejected attempt to reproduce scheduling.
    pub fn select_next(&mut self) -> Option<AgentId> {
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

/// Prototype runner with private, monotonic committed and attempted counters.
/// No durable write, snapshot/resume or final canonical event schema yet.
pub struct Engine<W: World, P: Policy<W>> {
    trial: TrialId,
    world: W,
    policy: P,
    schedule: RoundRobin,
    committed_step: u64,
    next_attempt: u64,
    last_event_id: Option<String>,
    events: Vec<CommittedEvent<W::Action>>,
    audit: Vec<AuditRecord<W::Action>>,
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
            committed_step: 0,
            next_attempt: 0,
            last_event_id: None,
            events: Vec::new(),
            audit: Vec::new(),
        })
    }

    pub fn world(&self) -> &W {
        &self.world
    }

    pub fn trial(&self) -> &TrialId {
        &self.trial
    }

    pub fn events(&self) -> &[CommittedEvent<W::Action>] {
        &self.events
    }

    pub fn audit(&self) -> &[AuditRecord<W::Action>] {
        &self.audit
    }

    pub fn committed_steps(&self) -> u64 {
        self.committed_step
    }

    pub fn attempted_steps(&self) -> u64 {
        self.next_attempt
    }

    /// Lifecycle changes will become logged commands in R1. For now this
    /// prototype cannot claim deterministic resume after pause/resume.
    pub fn pause_actor(&mut self, actor: AgentId) {
        self.schedule.pause(actor);
    }

    pub fn resume_actor(&mut self, actor: AgentId) {
        self.schedule.resume(actor);
    }

    /// Safely permit bounded export without resetting the logical step.
    /// Caller must persist drained events before discarding them.
    pub fn drain_events(&mut self) -> Vec<CommittedEvent<W::Action>> {
        std::mem::take(&mut self.events)
    }

    /// select -> observe -> propose -> validate/clone -> commit.
    /// Rejections DO advance the schedule and append an audit record but do
    /// NOT commit any world change or world event.
    pub fn step(&mut self) -> Result<Option<CommittedEvent<W::Action>>, EngineError> {
        let next_attempt = self
            .next_attempt
            .checked_add(1)
            .ok_or(EngineError::Exhausted)?;
        let next_committed = self
            .committed_step
            .checked_add(1)
            .ok_or(EngineError::Exhausted)?;
        let Some(actor) = self.schedule.select_next() else {
            return Ok(None);
        };
        let attempt = self.next_attempt;
        self.next_attempt = next_attempt;

        let outcome = (|| {
            let observation = self.world.observe(actor).map_err(EngineError::Observe)?;
            let action = self
                .policy
                .propose(actor, &observation)
                .map_err(EngineError::Policy)?;
            let mut staged_world = self.world.clone();
            staged_world
                .apply(actor, action.clone())
                .map_err(EngineError::Transition)?;
            let action_hash = action_digest(&action)?;
            Ok::<_, EngineError>((staged_world, action, action_hash))
        })();

        let (staged_world, action, action_hash) = match outcome {
            Ok(value) => value,
            Err(error) => {
                let reason = error.to_string();
                let rejection = RejectedAttempt {
                    schema_version: 2,
                    rejection_id: derive_id(
                        "rejection-v1",
                        &[
                            &self.trial.0,
                            &attempt.to_string(),
                            &actor.0.to_string(),
                            &reason,
                        ],
                    ),
                    trial_id: self.trial.0.clone(),
                    attempt_index: attempt,
                    next_logical_step: self.committed_step,
                    agent_id: actor,
                    reason,
                };
                self.audit.push(AuditRecord::Rejected(rejection));
                return Err(error);
            }
        };

        let step = self.committed_step;
        let event = CommittedEvent {
            schema_version: 2,
            event_id: derive_id(
                "event-v2",
                &[
                    &self.trial.0,
                    &step.to_string(),
                    &actor.0.to_string(),
                    &action_hash,
                    self.last_event_id.as_deref().unwrap_or(""),
                ],
            ),
            previous_event_id: self.last_event_id.clone(),
            action_hash,
            trial_id: self.trial.0.clone(),
            logical_step: step,
            agent_id: actor,
            action,
        };
        self.world = staged_world;
        self.committed_step = next_committed;
        self.last_event_id = Some(event.event_id.clone());
        self.events.push(event.clone());
        self.audit.push(AuditRecord::Committed(event.clone()));
        Ok(Some(event))
    }

    /// World-only JSONL export; the audit file includes rejected attempts.
    /// Neither export is durable append-to-log and neither is JCS canonical.
    pub fn write_jsonl(&self, mut output: impl Write) -> io::Result<()> {
        for event in &self.events {
            serde_json::to_writer(&mut output, event).map_err(io::Error::other)?;
            output.write_all(b"\n")?;
        }
        Ok(())
    }

    pub fn write_audit_jsonl(&self, mut output: impl Write) -> io::Result<()> {
        for record in &self.audit {
            serde_json::to_writer(&mut output, record).map_err(io::Error::other)?;
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
        reject_actor: Option<AgentId>,
        reject_observation: Option<AgentId>,
    }

    impl World for Counter {
        type Action = u64;
        type Observation = u64;

        fn observe(&self, actor: AgentId) -> Result<u64, String> {
            if self.reject_observation == Some(actor) {
                return Err("observation blocked".into());
            }
            Ok(self.value)
        }

        fn apply(&mut self, actor: AgentId, increment: u64) -> Result<(), String> {
            if self.reject_actor == Some(actor) {
                return Err("rejected".into());
            }
            self.value = self.value.checked_add(increment).ok_or("overflow")?;
            Ok(())
        }
    }

    struct Increment {
        amount: u64,
        fail_actor: Option<AgentId>,
    }

    impl Policy<Counter> for Increment {
        fn propose(&self, actor: AgentId, _: &u64) -> Result<u64, String> {
            if self.fail_actor == Some(actor) {
                return Err("policy blocked".into());
            }
            Ok(self.amount)
        }
    }

    fn engine(amount: u64) -> Engine<Counter, Increment> {
        Engine::new(
            TrialId::new("golden-two-agent", 7),
            Counter {
                value: 0,
                reject_actor: None,
                reject_observation: None,
            },
            Increment {
                amount,
                fail_actor: None,
            },
            [AgentId(0), AgentId(1)],
        )
        .unwrap()
    }

    #[test]
    fn scripted_golden_fixture_and_repeatability() {
        let fixture_text = include_str!("../fixtures/golden/scripted-two-agent-v1.json");
        let fixture: serde_json::Value = serde_json::from_str(fixture_text).unwrap();
        let mut first = engine(1);
        let mut second = engine(1);
        for _ in 0..4 {
            first.step().unwrap();
            second.step().unwrap();
        }
        let observed: Vec<u64> = first.events().iter().map(|e| e.agent_id.0).collect();
        let expected: Vec<u64> = fixture["expected_actor_order"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap())
            .collect();
        assert_eq!(observed, expected);
        assert_eq!(
            first.trial().0,
            fixture["expected_trial_id"].as_str().unwrap()
        );
        let expected_ids = fixture["expected_event_ids"].as_array().unwrap();
        for (event, expected_id) in first.events().iter().zip(expected_ids) {
            assert_eq!(event.event_id, expected_id.as_str().unwrap());
        }
        assert_eq!(
            first.world().value,
            fixture["expected_state"]["value"].as_u64().unwrap()
        );
        assert_eq!(first.events(), second.events());
        let mut a = Vec::new();
        let mut b = Vec::new();
        first.write_jsonl(&mut a).unwrap();
        second.write_jsonl(&mut b).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn failed_actor_does_not_starve_others_or_partially_commit_world() {
        let mut e = engine(1);
        e.world.reject_actor = Some(AgentId(0));
        assert_eq!(
            e.step().unwrap_err(),
            EngineError::Transition("rejected".into())
        );
        assert_eq!(e.world().value, 0);
        assert!(e.events().is_empty());
        assert_eq!(e.attempted_steps(), 1);
        assert_eq!(e.audit().len(), 1);
        assert!(matches!(&e.audit()[0], AuditRecord::Rejected(_)));
        let committed = e.step().unwrap().unwrap();
        assert_eq!(committed.agent_id, AgentId(1));
        assert_eq!(committed.logical_step, 0);
        assert_eq!(e.world().value, 1);
        assert_eq!(e.committed_steps(), 1);
        assert_eq!(e.audit().len(), 2);
        let mut trace = Vec::new();
        e.write_audit_jsonl(&mut trace).unwrap();
        assert_eq!(trace.iter().filter(|byte| **byte == b'\n').count(), 2);
    }

    #[test]
    fn observe_and_policy_failures_also_advance_scheduler() {
        let mut e = engine(1);
        e.world.reject_observation = Some(AgentId(0));
        assert!(matches!(e.step(), Err(EngineError::Observe(_))));
        assert_eq!(e.step().unwrap().unwrap().agent_id, AgentId(1));

        let mut e = engine(1);
        e.policy.fail_actor = Some(AgentId(0));
        assert!(matches!(e.step(), Err(EngineError::Policy(_))));
        assert_eq!(e.step().unwrap().unwrap().agent_id, AgentId(1));
    }

    #[test]
    fn duplicate_and_all_paused_have_defined_behavior() {
        assert!(matches!(
            RoundRobin::new([AgentId(3), AgentId(3)]),
            Err(EngineError::DuplicateAgent(AgentId(3)))
        ));
        let mut e = engine(1);
        e.pause_actor(AgentId(0));
        e.pause_actor(AgentId(1));
        assert!(e.step().unwrap().is_none());
        assert_eq!(e.attempted_steps(), 0);
        assert!(e.events().is_empty());
        e.resume_actor(AgentId(1));
        assert_eq!(e.step().unwrap().unwrap().agent_id, AgentId(1));
    }

    #[test]
    fn event_id_changes_for_different_action_and_after_drain_step_keeps_increasing() {
        let mut first = engine(1);
        let mut different = engine(5);
        let a = first.step().unwrap().unwrap();
        let b = different.step().unwrap().unwrap();
        assert_ne!(a.event_id, b.event_id);
        assert_ne!(a.action_hash, b.action_hash);

        let old = first.drain_events();
        assert_eq!(old.len(), 1);
        assert!(first.events().is_empty());
        let newer = first.step().unwrap().unwrap();
        assert_eq!(newer.logical_step, 1);
        assert_ne!(newer.event_id, old[0].event_id);
        assert_eq!(
            newer.previous_event_id.as_deref(),
            Some(old[0].event_id.as_str())
        );
    }

    #[test]
    fn ids_and_steps_serialize_as_decimal_strings() {
        let agent = AgentId(u64::MAX);
        let json = serde_json::to_string(&agent).unwrap();
        assert_eq!(json, format!("\"{}\"", u64::MAX));
        assert_eq!(serde_json::from_str::<AgentId>(&json).unwrap(), agent);
        assert!(serde_json::from_str::<AgentId>("123").is_err());
        assert!(serde_json::from_str::<AgentId>("\"0001\"").is_err());
        let event = CommittedEvent {
            schema_version: 2,
            event_id: "event".into(),
            previous_event_id: None,
            action_hash: "hash".into(),
            trial_id: "trial".into(),
            logical_step: u64::MAX,
            agent_id: agent,
            action: 1_u64,
        };
        let exported = serde_json::to_value(&event).unwrap();
        assert_eq!(exported["logical_step"], u64::MAX.to_string());
        assert_eq!(exported["agent_id"], u64::MAX.to_string());
    }

    #[test]
    fn action_hash_is_independent_of_object_insertion_order() {
        let mut left = serde_json::Map::new();
        left.insert("z".into(), serde_json::json!({"b": 2, "a": 1}));
        left.insert("a".into(), serde_json::json!([{"y": true, "x": false}]));
        let mut right = serde_json::Map::new();
        right.insert("a".into(), serde_json::json!([{"x": false, "y": true}]));
        right.insert("z".into(), serde_json::json!({"a": 1, "b": 2}));
        assert_eq!(
            action_digest(&serde_json::Value::Object(left)).unwrap(),
            action_digest(&serde_json::Value::Object(right)).unwrap()
        );
    }

    #[test]
    fn identities_are_length_prefixed_and_domain_separated() {
        assert_ne!(derive_id("x", &["ab", "c"]), derive_id("x", &["a", "bc"]));
        assert_ne!(TrialId::new("a", 1), TrialId::new("a", 2));
    }
}
