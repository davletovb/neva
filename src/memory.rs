//! Deterministic, bounded, application-owned structured memory.
//! Embed MemoryBank in a World to include it in that world's event log,
//! snapshots, replay, and forks. No embeddings, network, or hidden globals.
use crate::core::AgentId;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

pub const MEMORY_SCHEMA: u32 = 1;
pub const MAX_RECORDS: usize = 10_000;
pub const MAX_PAYLOAD_BYTES: usize = 64 * 1024;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryRecord {
    pub agent: AgentId,
    pub key: String,
    pub payload: Value,
    pub created_step: u64,
    pub accessed_step: u64,
    pub importance: u16,
    pub accesses: u32,
    /// Exclusive expiry step; None means durable.
    pub expires_at: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryBank {
    pub schema: u32,
    pub capacity: usize,
    pub max_payload_bytes: usize,
    /// Keys are unambiguous decimal agent IDs followed by ':' and a user key.
    records: BTreeMap<String, MemoryRecord>,
}

fn valid_key(key: &str) -> bool {
    !key.is_empty()
        && key.len() <= 128
        && key
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'/'))
}

fn record_id(agent: AgentId, key: &str) -> String {
    format!("{}:{key}", agent.0)
}

impl MemoryBank {
    pub fn new(capacity: usize, max_payload_bytes: usize) -> Result<Self, String> {
        if !(1..=MAX_RECORDS).contains(&capacity)
            || !(1..=MAX_PAYLOAD_BYTES).contains(&max_payload_bytes)
        {
            return Err("memory capacity or payload limit out of range".into());
        }
        Ok(Self {
            schema: MEMORY_SCHEMA,
            capacity,
            max_payload_bytes,
            records: BTreeMap::new(),
        })
    }

    /// Validate deserialized memory before accepting untrusted snapshots.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != MEMORY_SCHEMA
            || !(1..=MAX_RECORDS).contains(&self.capacity)
            || !(1..=MAX_PAYLOAD_BYTES).contains(&self.max_payload_bytes)
            || self.records.len() > self.capacity
        {
            return Err("invalid memory schema or limits".into());
        }
        for (id, record) in &self.records {
            if !valid_key(&record.key)
                || id != &record_id(record.agent, &record.key)
                || record.importance > 1000
                || record.accessed_step < record.created_step
                || record
                    .expires_at
                    .is_some_and(|expiry| expiry <= record.created_step)
                || serde_json::to_vec(&record.payload)
                    .map_err(|e| e.to_string())?
                    .len()
                    > self.max_payload_bytes
            {
                return Err("invalid memory record".into());
            }
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn prune(&mut self, now: u64) {
        self.records
            .retain(|_, value| value.expires_at.is_none_or(|expiry| now < expiry));
    }

    /// Upsert with deterministic LRU/importance eviction. Time is a logical
    /// simulation step, never system time. TTL is expressed in logical steps.
    pub fn put(
        &mut self,
        agent: AgentId,
        key: &str,
        payload: Value,
        now: u64,
        importance: u16,
        ttl: Option<u64>,
    ) -> Result<(), String> {
        if !valid_key(key) || importance > 1000 {
            return Err("invalid memory key or importance".into());
        }
        let bytes = serde_json::to_vec(&payload).map_err(|e| e.to_string())?;
        if bytes.len() > self.max_payload_bytes {
            return Err("memory payload exceeds configured limit".into());
        }
        let expires_at = match ttl {
            Some(0) => return Err("memory TTL must be positive".into()),
            Some(delta) => Some(now.checked_add(delta).ok_or("memory TTL overflow")?),
            None => None,
        };
        let id = record_id(agent, key);
        if let Some(prior) = self.records.get(&id) {
            if now < prior.accessed_step {
                return Err("memory logical clock went backwards".into());
            }
        }
        // Validate first; errors do not mutate or evict any records.
        self.prune(now);
        let created_step = self.records.get(&id).map_or(now, |r| r.created_step);
        let accesses = self.records.get(&id).map_or(0, |r| r.accesses);
        if !self.records.contains_key(&id) && self.records.len() == self.capacity {
            let victim = self
                .records
                .iter()
                .min_by_key(|(key, item)| (item.importance, item.accessed_step, *key))
                .map(|(key, _)| key.clone())
                .ok_or("memory is full but has no eviction candidate")?;
            self.records.remove(&victim);
        }
        self.records.insert(
            id,
            MemoryRecord {
                agent,
                key: key.to_owned(),
                payload,
                created_step,
                accessed_step: now,
                importance,
                accesses,
                expires_at,
            },
        );
        Ok(())
    }

    /// Exact-key recall; expired memories are not returned or reinforced.
    pub fn get(
        &mut self,
        agent: AgentId,
        key: &str,
        now: u64,
    ) -> Result<Option<MemoryRecord>, String> {
        if !valid_key(key) {
            return Err("invalid memory key".into());
        }
        self.prune(now);
        let Some(record) = self.records.get_mut(&record_id(agent, key)) else {
            return Ok(None);
        };
        if now < record.accessed_step {
            return Err("memory logical clock went backwards".into());
        }
        record.accessed_step = now;
        record.accesses = record.accesses.saturating_add(1);
        Ok(Some(record.clone()))
    }

    /// Stable ranking by importance, then use count, recent access, and key.
    /// Prefix recall is exact lexical matching, NOT vector/semantic recall.
    pub fn recall_prefix(
        &self,
        agent: AgentId,
        prefix: &str,
        now: u64,
        limit: usize,
    ) -> Result<Vec<MemoryRecord>, String> {
        if limit > self.capacity || prefix.len() > 128 {
            return Err("invalid recall limit or prefix".into());
        }
        let mut matches: Vec<_> = self
            .records
            .values()
            .filter(|r| {
                r.agent == agent
                    && r.key.starts_with(prefix)
                    && r.expires_at.is_none_or(|expiry| now < expiry)
            })
            .cloned()
            .collect();
        matches.sort_by(|a, b| {
            b.importance
                .cmp(&a.importance)
                .then_with(|| b.accesses.cmp(&a.accesses))
                .then_with(|| b.accessed_step.cmp(&a.accessed_step))
                .then_with(|| a.key.cmp(&b.key))
        });
        matches.truncate(limit);
        Ok(matches)
    }

    /// Explicit, deterministic adaptive salience adjustment (0..=1000).
    pub fn reinforce(
        &mut self,
        agent: AgentId,
        key: &str,
        now: u64,
        increment: u16,
    ) -> Result<bool, String> {
        if increment > 1000 || !valid_key(key) {
            return Err("invalid reinforcement".into());
        }
        self.prune(now);
        let Some(record) = self.records.get_mut(&record_id(agent, key)) else {
            return Ok(false);
        };
        if now < record.accessed_step {
            return Err("memory logical clock went backwards".into());
        }
        record.importance = record.importance.saturating_add(increment).min(1000);
        record.accessed_step = now;
        record.accesses = record.accesses.saturating_add(1);
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn deterministic_order_expiry_and_agent_separation() {
        let mut m = MemoryBank::new(3, 1024).unwrap();
        m.put(AgentId(1), "belief/a", json!({"v": 1}), 0, 10, Some(2))
            .unwrap();
        m.put(AgentId(2), "belief/a", json!({"v": 2}), 0, 10, None)
            .unwrap();
        m.put(AgentId(1), "belief/b", json!({"v": 3}), 1, 11, None)
            .unwrap();
        assert_eq!(
            m.recall_prefix(AgentId(1), "belief/", 1, 3).unwrap()[0].key,
            "belief/b"
        );
        assert!(m.get(AgentId(1), "belief/a", 2).unwrap().is_none());
        assert_eq!(
            m.get(AgentId(2), "belief/a", 2).unwrap().unwrap().payload,
            json!({"v":2})
        );
        m.validate().unwrap();
        let restored: MemoryBank =
            serde_json::from_slice(&serde_json::to_vec(&m).unwrap()).unwrap();
        assert_eq!(restored, m);
    }

    #[test]
    fn evicts_deterministically_and_reinforces() {
        let mut m = MemoryBank::new(2, 64).unwrap();
        m.put(AgentId(1), "a", json!(1), 0, 1, None).unwrap();
        m.put(AgentId(1), "b", json!(2), 0, 1, None).unwrap();
        assert!(m.reinforce(AgentId(1), "a", 1, 5).unwrap());
        m.put(AgentId(1), "c", json!(3), 2, 1, None).unwrap();
        assert!(m.get(AgentId(1), "b", 2).unwrap().is_none());
        assert!(m.get(AgentId(1), "a", 2).unwrap().is_some());
    }

    #[test]
    fn invalid_upsert_does_not_mutate() {
        let mut m = MemoryBank::new(1, 4).unwrap();
        m.put(AgentId(0), "a", json!(1), 2, 1, None).unwrap();
        let original = m.clone();
        assert!(
            m.put(AgentId(0), "b", json!("oversized"), 4, 1, None)
                .is_err()
        );
        assert!(m.put(AgentId(0), "a", json!(2), 1, 1, None).is_err());
        assert!(m.put(AgentId(0), "a", json!(2), 4, 1, Some(0)).is_err());
        assert_eq!(m, original);
        let mut invalid = m.clone();
        invalid.capacity = 0;
        assert!(invalid.validate().is_err());
    }
}
