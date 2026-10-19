//! Trial-owned named ChaCha20 streams with persisted exact word positions.
//! The algorithm and seed-derivation profile are versioned; no process RNG.

use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Streams {
    pub profile: String,
    pub trial: String,
    #[serde(with = "crate::core::decimal_u64")]
    pub root_seed: u64,
    /// Full ChaCha20 word offset, decimal text because it is u128.
    pub positions: BTreeMap<String, String>,
    /// Shared RNG origin for paired branches; branch identity remains distinct.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed_domain: Option<String>,
}

impl Streams {
    pub fn new(trial: String, seed: u64) -> Self {
        Self {
            profile: "chacha20-0.9-words-v1".into(),
            trial,
            root_seed: seed,
            positions: BTreeMap::new(),
            seed_domain: None,
        }
    }

    fn stream_seed(&self, name: &str) -> [u8; 32] {
        let mut sha = Sha256::new();
        sha.update(b"hiyal:rng:v1");
        for field in [
            self.seed_domain.as_deref().unwrap_or(&self.trial),
            &self.root_seed.to_string(),
            name,
        ] {
            sha.update((field.len() as u64).to_be_bytes());
            sha.update(field.as_bytes());
        }
        sha.finalize().into()
    }

    /// Branch with a different identity but the same source random streams
    /// and exact word positions (common random numbers).
    pub fn paired_branch(&self, branch_trial_id: String) -> Self {
        let mut next = self.clone();
        next.seed_domain = Some(
            self.seed_domain.clone().unwrap_or_else(|| self.trial.clone())
        );
        next.trial = branch_trial_id;
        next
    }

    pub fn draw_u64(&mut self, name: &str) -> Result<u64, String> {
        if self.profile != "chacha20-0.9-words-v1" {
            return Err("unknown RNG profile".into());
        }
        let position: u128 = self
            .positions
            .get(name)
            .map(|n| n.parse::<u128>())
            .transpose()
            .map_err(|_| "invalid RNG position")?
            .unwrap_or(0);
        let mut rng = ChaCha20Rng::from_seed(self.stream_seed(name));
        rng.set_word_pos(position);
        let value = rng.next_u64();
        self.positions
            .insert(name.to_owned(), rng.get_word_pos().to_string());
        Ok(value)
    }

    /// Exact uniform choice without modulo bias. n must be positive.
    pub fn uniform_index(&mut self, stream: &str, n: usize) -> Result<usize, String> {
        let bound = u64::try_from(n).map_err(|_| "scheduler size exceeds u64")?;
        if bound == 0 {
            return Err("cannot sample empty population".into());
        }
        let rejection_limit = bound.wrapping_neg() % bound;
        loop {
            let draw = self.draw_u64(stream)?;
            if draw >= rejection_limit {
                return Ok((draw % bound) as usize);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn named_streams_restore_without_interference() {
        let mut a = Streams::new("trial".into(), 17);
        let first = a.draw_u64("scheduler").unwrap();
        let snapshot = a.clone();
        a.draw_u64("actor/1").unwrap();
        let next = a.draw_u64("scheduler").unwrap();
        let mut restored = snapshot;
        assert_eq!(next, restored.draw_u64("scheduler").unwrap());
        assert_ne!(first, next);
        assert_ne!(
            Streams::new("trial".into(), 18)
                .draw_u64("scheduler")
                .unwrap(),
            first
        );
    }
    #[test]
    fn paired_branches_share_random_numbers_but_not_identity() {
        let mut parent = Streams::new("parent".into(), 123);
        let _ = parent.draw_u64("scheduler").unwrap();
        let mut a = parent.paired_branch("control".into());
        let mut b = parent.paired_branch("treatment".into());
        assert_ne!(a.trial, b.trial);
        for _ in 0..10 {
            assert_eq!(a.draw_u64("scheduler").unwrap(), b.draw_u64("scheduler").unwrap());
        }
        assert_eq!(a.seed_domain.as_deref(), Some("parent"));
    }

    #[test]
    fn bounded_sampling_is_valid() {
        let mut streams = Streams::new("t".into(), 9);
        for _ in 0..3000 {
            assert!(streams.uniform_index("s", 3).unwrap() < 3);
        }
        assert!(streams.uniform_index("s", 0).is_err());
    }
}
