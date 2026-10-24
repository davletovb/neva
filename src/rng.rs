//! Trial-owned named ChaCha20 streams with persisted exact word positions.
//! The algorithm and seed-derivation profile are versioned; no process RNG.

use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap},
};

/// Overwrite `text` with the decimal `position`, reusing its buffer.
fn write_position(text: &mut String, position: u128) {
    use std::fmt::Write;
    text.clear();
    write!(text, "{position}").expect("writing to a String cannot fail");
}

/// Seed derivations keyed by (seed domain, root seed, stream name). The seed
/// is a pure function of exactly these inputs, so a hit returns the same
/// bytes `derive_stream_seed` would; the cache only skips the SHA-256.
type SeedCache = HashMap<String, HashMap<u64, HashMap<String, [u8; 32]>>>;
const SEED_CACHE_LIMIT: usize = 4096;
thread_local! {
    static SEED_CACHE: RefCell<(usize, SeedCache)> = RefCell::new((0, HashMap::new()));
}

fn derive_stream_seed(domain: &str, root_seed: u64, name: &str) -> [u8; 32] {
    let mut sha = Sha256::new();
    sha.update(b"hiyal:rng:v1");
    for field in [domain, &root_seed.to_string(), name] {
        sha.update((field.len() as u64).to_be_bytes());
        sha.update(field.as_bytes());
    }
    sha.finalize().into()
}

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
    /// The generator of the stream drawn last, kept positioned so a run of
    /// draws on one stream does not re-key ChaCha20 for every word. Not part
    /// of the state: never serialized, ignored by `==`, empty in a clone.
    #[serde(skip)]
    last: LastStream,
}

/// The most recently drawn stream: its seed inputs, its generator, and the
/// position text that draw wrote to `positions`. `draw_u64` uses it only
/// when all of these still match, so any change made to `Streams` between
/// draws (a position, the seed, the domain) falls back to re-keying.
#[derive(Default)]
struct LastStream(Option<Box<Positioned>>);

struct Positioned {
    domain: String,
    root_seed: u64,
    name: String,
    position: String,
    rng: ChaCha20Rng,
}

impl Clone for LastStream {
    fn clone(&self) -> Self {
        Self(None)
    }
}
impl PartialEq for LastStream {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}
impl Eq for LastStream {}
impl std::fmt::Debug for LastStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LastStream")
    }
}

impl Streams {
    pub fn new(trial: String, seed: u64) -> Self {
        Self {
            profile: "chacha20-0.9-words-v1".into(),
            trial,
            root_seed: seed,
            positions: BTreeMap::new(),
            seed_domain: None,
            last: LastStream::default(),
        }
    }

    fn stream_seed(&self, name: &str) -> [u8; 32] {
        let domain = self.seed_domain.as_deref().unwrap_or(&self.trial);
        SEED_CACHE.with(|cache| {
            let (entries, cache) = &mut *cache.borrow_mut();
            if let Some(seed) = cache
                .get(domain)
                .and_then(|by_root| by_root.get(&self.root_seed))
                .and_then(|by_name| by_name.get(name))
            {
                return *seed;
            }
            if *entries >= SEED_CACHE_LIMIT {
                cache.clear();
                *entries = 0;
            }
            let seed = derive_stream_seed(domain, self.root_seed, name);
            cache
                .entry(domain.to_owned())
                .or_default()
                .entry(self.root_seed)
                .or_default()
                .insert(name.to_owned(), seed);
            *entries += 1;
            seed
        })
    }

    /// Branch with a different identity but the same source random streams
    /// and exact word positions (common random numbers).
    pub fn paired_branch(&self, branch_trial_id: String) -> Self {
        let mut next = self.clone();
        next.seed_domain = Some(
            self.seed_domain
                .clone()
                .unwrap_or_else(|| self.trial.clone()),
        );
        next.trial = branch_trial_id;
        next
    }

    pub fn draw_u64(&mut self, name: &str) -> Result<u64, String> {
        if self.profile != "chacha20-0.9-words-v1" {
            return Err("unknown RNG profile".into());
        }
        let domain = self.seed_domain.as_deref().unwrap_or(&self.trial);
        if let Some(last) = self.last.0.as_deref_mut()
            && last.name == name
            && last.root_seed == self.root_seed
            && last.domain == domain
            && let Some(position) = self.positions.get_mut(name)
            && *position == last.position
        {
            // The same generator `from_seed` + `set_word_pos` would build,
            // already at this word: same value, same next position.
            let value = last.rng.next_u64();
            write_position(position, last.rng.get_word_pos());
            last.position.clone_from(position);
            return Ok(value);
        }
        // Re-key: derive the seed, build and position the generator. The
        // cache's buffers are reused, so a stream switch allocates nothing
        // beyond what the position bookkeeping always did.
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
        let text = rng.get_word_pos().to_string();
        let domain = self.seed_domain.as_deref().unwrap_or(&self.trial);
        match self.last.0.as_deref_mut() {
            Some(last) => {
                last.domain.clear();
                last.domain.push_str(domain);
                last.root_seed = self.root_seed;
                last.name.clear();
                last.name.push_str(name);
                last.position.clone_from(&text);
                last.rng = rng;
            }
            None => {
                self.last = LastStream(Some(Box::new(Positioned {
                    domain: domain.to_owned(),
                    root_seed: self.root_seed,
                    name: name.to_owned(),
                    position: text.clone(),
                    rng,
                })))
            }
        }
        self.positions.insert(name.to_owned(), text);
        Ok(value)
    }

    /// Validate that policy/world draws only advance named streams. The
    /// scheduler's stream must be left to the engine alone.
    pub fn checked_progress(&self, from: &Self) -> Result<(), String> {
        if self.profile != from.profile
            || self.trial != from.trial
            || self.root_seed != from.root_seed
            || self.seed_domain != from.seed_domain
            || self.positions.get("scheduler") != from.positions.get("scheduler")
        {
            return Err("RNG origin or scheduler stream modified outside scheduler".into());
        }
        for (name, before) in &from.positions {
            let old = before
                .parse::<u128>()
                .map_err(|_| "invalid old RNG position")?;
            let new = self
                .positions
                .get(name)
                .ok_or("removed named RNG stream")?
                .parse::<u128>()
                .map_err(|_| "invalid new RNG position")?;
            if old > new || before != &old.to_string() {
                return Err("non-monotonic RNG stream".into());
            }
        }
        for value in self.positions.values() {
            let parsed = value.parse::<u128>().map_err(|_| "invalid RNG position")?;
            if value != &parsed.to_string() {
                return Err("noncanonical RNG position".into());
            }
        }
        Ok(())
    }

    /// Enforce that an agent callback advances only its own named stream
    /// (or a reducer only the world family). Other streams cannot be added,
    /// removed or modified by this callback.
    pub fn checked_scope_progress(&self, before: &Self, family: &str) -> Result<(), String> {
        self.checked_progress(before)?;
        for (name, position) in &self.positions {
            let allowed = name == family
                || name
                    .strip_prefix(family)
                    .is_some_and(|s| s.starts_with('/'));
            if !allowed && before.positions.get(name) != Some(position) {
                return Err(format!("unauthorized RNG stream draw: {name}"));
            }
        }
        Ok(())
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
            assert_eq!(
                a.draw_u64("scheduler").unwrap(),
                b.draw_u64("scheduler").unwrap()
            );
        }
        assert_eq!(a.seed_domain.as_deref(), Some("parent"));
    }

    #[test]
    fn pinned_chacha20_golden_and_actor_stream_isolation() {
        // Derived independently from SHA-256 seed domains + the ChaCha20
        // initial block. A dependency upgrade must not silently change D1.
        let mut s = Streams::new("trial".into(), 17);
        assert_eq!(s.draw_u64("scheduler").unwrap(), 9758413615208853506u64);
        let scheduled = s.clone();
        let _ = s.draw_u64("agent/2").unwrap();
        s.checked_progress(&scheduled).unwrap();
        let old = s.clone();
        assert!(s.draw_u64("scheduler").is_ok());
        assert!(s.checked_progress(&old).is_err());
    }

    #[test]
    fn cached_seed_matches_uncached_derivation() {
        let mut s = Streams::new("trial".into(), 17).paired_branch("branch".into());
        for name in ["scheduler", "agent/0", "world", "scheduler"] {
            assert_eq!(s.stream_seed(name), derive_stream_seed("trial", 17, name));
        }
        s.root_seed = 18;
        assert_eq!(
            s.stream_seed("world"),
            derive_stream_seed("trial", 18, "world")
        );
        s.seed_domain = None;
        assert_eq!(
            s.stream_seed("world"),
            derive_stream_seed("branch", 18, "world")
        );
    }

    #[test]
    fn bounded_sampling_is_valid() {
        let mut streams = Streams::new("t".into(), 9);
        for _ in 0..3000 {
            assert!(streams.uniform_index("s", 3).unwrap() < 3);
        }
        assert!(streams.uniform_index("s", 0).is_err());
    }

    /// `draw_u64` before the generator cache, deriving the seed without the
    /// seed cache either: the reference the fast path must match exactly.
    fn reference_draw(s: &mut Streams, name: &str) -> Result<u64, String> {
        if s.profile != "chacha20-0.9-words-v1" {
            return Err("unknown RNG profile".into());
        }
        let position: u128 = s
            .positions
            .get(name)
            .map(|n| n.parse::<u128>())
            .transpose()
            .map_err(|_| "invalid RNG position")?
            .unwrap_or(0);
        let domain = s.seed_domain.as_deref().unwrap_or(&s.trial);
        let mut rng = ChaCha20Rng::from_seed(derive_stream_seed(domain, s.root_seed, name));
        rng.set_word_pos(position);
        let value = rng.next_u64();
        s.positions
            .insert(name.to_owned(), rng.get_word_pos().to_string());
        Ok(value)
    }

    fn same_state(a: &Streams, b: &Streams) {
        assert_eq!(a, b);
        assert_eq!(
            serde_json::to_string(a).unwrap(),
            serde_json::to_string(b).unwrap()
        );
    }

    #[test]
    fn cached_generator_matches_rekeyed_draws_under_any_interleaving() {
        // `fast` uses draw_u64; `slow` mirrors every operation with the
        // reference draw. Outputs and full state must agree at every step.
        let names = [
            "setup/graph",
            "agent/0/selection",
            "agent/1/selection",
            "world",
        ];
        let mut script = ChaCha20Rng::seed_from_u64(1729);
        let mut fast = Streams::new("trial".into(), 72100);
        let mut slow = fast.clone();
        for step in 0..20_000 {
            let name = if script.next_u32() % 10 < 7 {
                names[0] // mostly long runs on one stream, as in setup
            } else {
                names[(script.next_u32() % 4) as usize]
            };
            match script.next_u32() % 100 {
                0 => {
                    // Continue on a clone (empty cache) of the fast side.
                    fast = fast.clone();
                }
                1 => {
                    // Rewind or skip a position from outside.
                    let to = (script.next_u32() % 50).to_string();
                    fast.positions.insert(name.into(), to.clone());
                    slow.positions.insert(name.into(), to);
                }
                2 => {
                    fast.positions.remove(name);
                    slow.positions.remove(name);
                }
                3 => {
                    // Non-canonical text the reference still parses.
                    let to = format!("00{}", script.next_u32() % 50);
                    fast.positions.insert(name.into(), to.clone());
                    slow.positions.insert(name.into(), to);
                }
                4 => {
                    fast.root_seed ^= 1;
                    slow.root_seed ^= 1;
                }
                5 => {
                    let branch = format!("branch-{step}");
                    fast = fast.paired_branch(branch.clone());
                    slow = slow.paired_branch(branch);
                }
                6 => {
                    fast = serde_json::from_str(&serde_json::to_string(&fast).unwrap()).unwrap();
                }
                8 => {
                    // Change the seed domain in place, without a clone.
                    let domain = format!("domain-{}", script.next_u32() % 3);
                    fast.seed_domain = Some(domain.clone());
                    slow.seed_domain = Some(domain);
                }
                9 => {
                    let trial = format!("trial-{}", script.next_u32() % 3);
                    fast.trial = trial.clone();
                    slow.trial = trial;
                    if script.next_u32() % 2 == 0 {
                        fast.seed_domain = None;
                        slow.seed_domain = None;
                    }
                }
                7 => {
                    let bound = 1 + (script.next_u32() % 90) as usize;
                    let got = fast.uniform_index(name, bound).unwrap();
                    // uniform_index is built on draw_u64; mirror it.
                    let limit = (bound as u64).wrapping_neg() % bound as u64;
                    let expected = loop {
                        let draw = reference_draw(&mut slow, name).unwrap();
                        if draw >= limit {
                            break (draw % bound as u64) as usize;
                        }
                    };
                    assert_eq!(got, expected, "uniform_index at step {step}");
                }
                _ => {}
            }
            assert_eq!(
                fast.draw_u64(name).unwrap(),
                reference_draw(&mut slow, name).unwrap(),
                "draw at step {step} on {name}"
            );
            same_state(&fast, &slow);
        }
    }

    #[test]
    fn a_clone_drawing_on_shares_no_generator_with_its_original() {
        let mut a = Streams::new("trial".into(), 5);
        a.draw_u64("x").unwrap();
        let mut b = a.clone();
        let from_b: Vec<_> = (0..5).map(|_| b.draw_u64("x").unwrap()).collect();
        let from_a: Vec<_> = (0..5).map(|_| a.draw_u64("x").unwrap()).collect();
        assert_eq!(
            from_a, from_b,
            "both continue the same stream independently"
        );
        same_state(&a, &b);
    }

    #[test]
    fn the_generator_cache_is_not_part_of_the_state() {
        let mut drawn = Streams::new("trial".into(), 9);
        for _ in 0..3 {
            drawn.draw_u64("x").unwrap();
        }
        let json = serde_json::to_string(&drawn).unwrap();
        assert!(!json.contains("last"), "{json}");
        let restored: Streams = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, drawn);
        assert!(restored.last.0.is_none() && drawn.last.0.is_some());
    }

    #[test]
    fn an_invalid_position_is_refused_even_with_a_cached_generator() {
        let mut s = Streams::new("trial".into(), 9);
        s.draw_u64("x").unwrap();
        for bad in ["", "not a number", "-2", "1e3"] {
            s.positions.insert("x".into(), bad.into());
            assert_eq!(
                s.draw_u64("x"),
                Err("invalid RNG position".into()),
                "{bad:?}"
            );
            assert_eq!(s.positions["x"], bad, "a refused draw leaves the position");
        }
    }

    #[test]
    fn cached_draws_cross_chacha_block_boundaries_like_rekeyed_ones() {
        // A ChaCha20 block is 16 words, and the generator buffers 4 blocks
        // (64 words, 32 draws). Start just before each kind of boundary,
        // including one where a draw straddles a block (odd word offset).
        for start in [
            14u128,
            15,
            30,
            31,
            62,
            63,
            64,
            127,
            (1 << 32) - 1,
            (1 << 68) - 3,
        ] {
            let mut fast = Streams::new("trial".into(), 11);
            fast.positions.insert("x".into(), start.to_string());
            let mut slow = fast.clone();
            for draw in 0..80 {
                assert_eq!(
                    fast.draw_u64("x").unwrap(),
                    reference_draw(&mut slow, "x").unwrap(),
                    "start {start}, draw {draw}"
                );
                same_state(&fast, &slow);
            }
        }
    }

    #[test]
    fn an_unknown_profile_is_refused_even_with_a_cached_generator() {
        let mut s = Streams::new("trial".into(), 9);
        s.draw_u64("x").unwrap();
        s.profile = "other".into();
        assert!(s.draw_u64("x").is_err());
    }
}
