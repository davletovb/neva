//! Deterministic, bounded, application-owned structured memory.
//! Embed MemoryBank in a World to include it in event logs and snapshots.
//! No embeddings, network, or global mutation.
use crate::core::AgentId;
use serde::{Deserialize, Deserializer, Serialize, de::Error as _};
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
    /// Exclusive expiry step. None means durable.
    pub expires_at: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MemoryBank {
    schema: u32,
    capacity: usize,
    max_payload_bytes: usize,
    /// All mutating and read operations use one monotonic logical clock.
    last_step: u64,
    /// Keys: canonical decimal agent ID + ':' + user key.
    records: BTreeMap<String, MemoryRecord>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MemoryBankWire {
    schema: u32,
    capacity: usize,
    max_payload_bytes: usize,
    last_step: u64,
    records: BTreeMap<String, MemoryRecord>,
}

impl<'de> Deserialize<'de> for MemoryBank {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let data = MemoryBankWire::deserialize(deserializer)?;
        let bank = Self {
            schema: data.schema,
            capacity: data.capacity,
            max_payload_bytes: data.max_payload_bytes,
            last_step: data.last_step,
            records: data.records,
        };
        bank.validate().map_err(D::Error::custom)?;
        Ok(bank)
    }
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
            last_step: 0,
            records: BTreeMap::new(),
        })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn max_payload_bytes(&self) -> usize {
        self.max_payload_bytes
    }
    pub fn last_step(&self) -> u64 {
        self.last_step
    }
    pub fn len(&self) -> usize {
        self.records.len()
    }
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Used by Deserialize, or manually after application-owned migrations.
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
                || record.created_step > record.accessed_step
                || record.accessed_step > self.last_step
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

    fn check_time(&self, now: u64) -> Result<(), String> {
        if now < self.last_step {
            Err("memory logical clock went backwards".into())
        } else {
            Ok(())
        }
    }

    /// Explicit O(n) maintenance. Ordinary lookups never scan all agents.
    pub fn prune(&mut self, now: u64) -> Result<usize, String> {
        self.check_time(now)?;
        let previous = self.records.len();
        self.records
            .retain(|_, record| record.expires_at.is_none_or(|expiry| now < expiry));
        self.last_step = now;
        Ok(previous - self.records.len())
    }

    /// Upsert with deterministic importance/LRU eviction. TTL uses logical steps.
    /// Validation happens before mutation. Full-bank pruning runs only on
    /// insertion at the capacity boundary.
    pub fn put(
        &mut self,
        agent: AgentId,
        key: &str,
        payload: Value,
        now: u64,
        importance: u16,
        ttl: Option<u64>,
    ) -> Result<(), String> {
        self.check_time(now)?;
        if !valid_key(key) || importance > 1000 {
            return Err("invalid memory key or importance".into());
        }
        if serde_json::to_vec(&payload)
            .map_err(|e| e.to_string())?
            .len()
            > self.max_payload_bytes
        {
            return Err("memory payload exceeds configured limit".into());
        }
        let expires_at = match ttl {
            Some(0) => return Err("memory TTL must be positive".into()),
            Some(delta) => Some(now.checked_add(delta).ok_or("memory TTL overflow")?),
            None => None,
        };
        let id = record_id(agent, key);
        // Old expired entries are not "updates": restart their creation time.
        if self
            .records
            .get(&id)
            .is_some_and(|r| r.expires_at.is_some_and(|expiry| now >= expiry))
        {
            self.records.remove(&id);
        }
        if !self.records.contains_key(&id) && self.records.len() >= self.capacity {
            self.prune(now)?;
            while self.records.len() >= self.capacity {
                let victim = self
                    .records
                    .iter()
                    .min_by_key(|(key, r)| (r.importance, r.accessed_step, *key))
                    .map(|(key, _)| key.clone())
                    .ok_or("memory has no eviction candidate")?;
                self.records.remove(&victim);
            }
        }
        let created_step = self.records.get(&id).map_or(now, |r| r.created_step);
        let accesses = self.records.get(&id).map_or(0, |r| r.accesses);
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
        self.last_step = now;
        Ok(())
    }

    /// Exact-key recall; expired records are removed lazily on touch.
    pub fn get(
        &mut self,
        agent: AgentId,
        key: &str,
        now: u64,
    ) -> Result<Option<MemoryRecord>, String> {
        self.check_time(now)?;
        if !valid_key(key) {
            return Err("invalid memory key".into());
        }
        let id = record_id(agent, key);
        if self
            .records
            .get(&id)
            .is_some_and(|r| r.expires_at.is_some_and(|expiry| now >= expiry))
        {
            self.records.remove(&id);
        }
        self.last_step = now;
        let Some(record) = self.records.get_mut(&id) else {
            return Ok(None);
        };
        record.accessed_step = now;
        record.accesses = record.accesses.saturating_add(1);
        Ok(Some(record.clone()))
    }

    /// Stable ranking by importance, use count, recency and key. Prefix
    /// filtering uses the agent-key BTreeMap range; only results are cloned.
    pub fn recall_prefix(
        &mut self,
        agent: AgentId,
        prefix: &str,
        now: u64,
        limit: usize,
    ) -> Result<Vec<MemoryRecord>, String> {
        self.check_time(now)?;
        if limit > self.capacity || prefix.len() > 128 {
            return Err("invalid recall limit or prefix".into());
        }
        // A read at a future step advances the bank-wide logical clock,
        // including empty results and zero-limit queries.
        self.last_step = now;
        if limit == 0 {
            return Ok(Vec::new());
        }
        let seek = record_id(agent, prefix);
        let mut matches: Vec<_> = self
            .records
            .range(seek.clone()..)
            .take_while(|(id, _)| id.starts_with(&seek))
            .map(|(_, record)| record)
            .filter(|r| r.expires_at.is_none_or(|expiry| now < expiry))
            .collect();
        matches.sort_by(|a, b| {
            b.importance
                .cmp(&a.importance)
                .then_with(|| b.accesses.cmp(&a.accesses))
                .then_with(|| b.accessed_step.cmp(&a.accessed_step))
                .then_with(|| a.key.cmp(&b.key))
        });
        Ok(matches.into_iter().take(limit).cloned().collect())
    }

    /// Adaptive importance: explicit, saturating reinforcement.
    pub fn reinforce(
        &mut self,
        agent: AgentId,
        key: &str,
        now: u64,
        increment: u16,
    ) -> Result<bool, String> {
        self.check_time(now)?;
        if increment > 1000 || !valid_key(key) {
            return Err("invalid reinforcement".into());
        }
        let id = record_id(agent, key);
        if self
            .records
            .get(&id)
            .is_some_and(|r| r.expires_at.is_some_and(|expiry| now >= expiry))
        {
            self.records.remove(&id);
        }
        self.last_step = now;
        let Some(record) = self.records.get_mut(&id) else {
            return Ok(false);
        };
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
    fn invalid_calls_leave_bank_unchanged_and_clock_is_global() {
        let mut m = MemoryBank::new(3, 32).unwrap();
        m.put(AgentId(1), "expires", json!(1), 10, 1, Some(2))
            .unwrap();
        m.put(AgentId(2), "new", json!(2), 10, 1, None).unwrap();
        let prior = m.clone();
        assert!(m.get(AgentId(1), "expires", 9).is_err());
        assert!(m.reinforce(AgentId(1), "expires", 9, 2).is_err());
        assert!(m.put(AgentId(3), "new", json!(3), 9, 1, None).is_err());
        assert!(m.prune(9).is_err());
        assert!(
            m.put(AgentId(1), "b", json!("oversized ".repeat(6)), 12, 1, None)
                .is_err()
        );
        assert_eq!(m, prior);
        assert_eq!(m.get(AgentId(1), "expires", 12).unwrap(), None);
    }

    #[test]
    fn untrusted_deserialization_rejects_malformed_limits_and_clocks() {
        let m = MemoryBank::new(1, 32).unwrap();
        let mut wire = serde_json::to_value(&m).unwrap();
        wire["capacity"] = json!(0);
        assert!(serde_json::from_value::<MemoryBank>(wire).is_err());
        let mut m = m;
        m.put(AgentId(1), "x", json!(1), 10, 1, None).unwrap();
        let mut wire = serde_json::to_value(&m).unwrap();
        wire["last_step"] = json!(5);
        assert!(serde_json::from_value::<MemoryBank>(wire).is_err());
    }

    #[test]
    fn prefix_recall_advances_clock_even_when_empty_or_zero_limit() {
        let mut m = MemoryBank::new(3, 32).unwrap();
        m.put(AgentId(1), "expires", json!(1), 1, 1, Some(2))
            .unwrap();
        assert!(
            m.recall_prefix(AgentId(1), "expires", 3, 1)
                .unwrap()
                .is_empty()
        );
        let prior = m.clone();
        assert!(m.get(AgentId(1), "expires", 2).is_err());
        assert_eq!(m, prior);
        assert!(m.recall_prefix(AgentId(1), "", 4, 0).unwrap().is_empty());
        assert_eq!(m.last_step(), 4);
        assert!(m.get(AgentId(1), "expires", 3).is_err());
    }

    #[test]
    fn scoped_prefix_does_not_match_other_agents_or_clone_all() {
        let mut m = MemoryBank::new(10, 32).unwrap();
        m.put(AgentId(1), "p/a", json!(1), 0, 1, None).unwrap();
        m.put(AgentId(11), "p/b", json!(2), 0, 100, None).unwrap();
        m.put(AgentId(1), "p/c", json!(3), 0, 2, None).unwrap();
        assert_eq!(
            m.recall_prefix(AgentId(1), "p/", 0, 1).unwrap()[0].key,
            "p/c"
        );
        assert!(m.recall_prefix(AgentId(1), "p/b", 0, 1).unwrap().is_empty());
    }
}
