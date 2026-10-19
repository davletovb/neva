//! Deterministic round-robin and seeded random selection with bounded probes.
use crate::core::AgentId;
use crate::rng::Streams;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scheduling {
    RoundRobin,
    Random,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Schedule {
    pub mode: Scheduling,
    pub agents: Vec<AgentId>,
    pub paused: BTreeSet<AgentId>,
    pub cursor: usize,
}

impl Schedule {
    pub fn new(mode: Scheduling, agents: Vec<AgentId>) -> Result<Self, String> {
        if agents.is_empty() {
            return Err("at least one actor required".into());
        }
        let unique: BTreeSet<_> = agents.iter().copied().collect();
        if unique.len() != agents.len() {
            return Err("duplicate agent ID".into());
        }
        Ok(Self {
            mode,
            agents,
            paused: BTreeSet::new(),
            cursor: 0,
        })
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.agents.is_empty() || self.cursor >= self.agents.len() {
            return Err("invalid scheduler population/cursor".into());
        }
        let unique: BTreeSet<_> = self.agents.iter().copied().collect();
        if unique.len() != self.agents.len() || !self.paused.is_subset(&unique) {
            return Err("duplicate or unknown paused actor in scheduler".into());
        }
        Ok(())
    }

    pub fn select(&mut self, streams: &mut Streams) -> Result<Option<AgentId>, String> {
        self.validate()?;
        if self.agents.iter().all(|a| self.paused.contains(a)) {
            return Ok(None);
        }
        match self.mode {
            Scheduling::RoundRobin => {
                for _ in 0..self.agents.len() {
                    let actor = self.agents[self.cursor];
                    self.cursor = (self.cursor + 1) % self.agents.len();
                    if !self.paused.contains(&actor) {
                        return Ok(Some(actor));
                    }
                }
                Ok(None)
            }
            Scheduling::Random => {
                let candidates: Vec<_> = self
                    .agents
                    .iter()
                    .filter(|a| !self.paused.contains(a))
                    .copied()
                    .collect();
                let idx = streams.uniform_index("scheduler", candidates.len())?;
                Ok(Some(candidates[idx]))
            }
        }
    }
    pub fn pause(&mut self, actor: AgentId) -> Result<(), String> {
        if !self.agents.contains(&actor) {
            return Err("unknown actor".into());
        }
        self.paused.insert(actor);
        Ok(())
    }
    pub fn resume(&mut self, actor: AgentId) -> Result<(), String> {
        if !self.agents.contains(&actor) {
            return Err("unknown actor".into());
        }
        self.paused.remove(&actor);
        Ok(())
    }
}
#[cfg(test)]
mod invariant_tests {
    use super::*;

    #[test]
    fn reject_corrupt_persisted_schedules_before_indexing() {
        let mut s = Schedule::new(Scheduling::RoundRobin, vec![AgentId(0), AgentId(1)]).unwrap();
        s.cursor = 7;
        assert!(s.validate().is_err());
        s.cursor = 0;
        s.paused.insert(AgentId(123));
        assert!(s.validate().is_err());
        s.paused.clear();
        s.agents.push(AgentId(1));
        assert!(s.validate().is_err());
    }
}
