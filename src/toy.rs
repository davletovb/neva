//! Tiny independently checkable reference societies for R2 verification.
//! These are scientific *toy* models, not empirical sociological findings.
use crate::{
    core::{AgentId, Policy, World},
    rng::Streams,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Society {
    pub balances: Vec<u64>,
    pub degrees: Vec<u64>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToyAction {
    Transfer { from: usize, to: usize, amount: u64 },
    Attach { from: usize, to: usize },
}
#[derive(Clone)]
pub struct Observation {
    pub balances: Vec<u64>,
}
impl World for Society {
    type Action = ToyAction;
    type Observation = Observation;
    fn observe(&self, _: AgentId) -> Result<Observation, String> {
        Ok(Observation {
            balances: self.balances.clone(),
        })
    }
    fn apply(&mut self, actor: AgentId, action: ToyAction) -> Result<(), String> {
        if self.balances.len() < 2 || self.degrees.len() != self.balances.len() {
            return Err("invalid toy world".into());
        }
        let id = usize::try_from(actor.0).map_err(|e| e.to_string())?;
        match action {
            ToyAction::Transfer { from, to, amount } => {
                if id != from || to >= self.balances.len() || from == to {
                    return Err("invalid transfer actor/target".into());
                }
                let next_from = self.balances[from]
                    .checked_sub(amount)
                    .ok_or("insufficient funds")?;
                let next_to = self.balances[to]
                    .checked_add(amount)
                    .ok_or("balance overflow")?;
                self.balances[from] = next_from;
                self.balances[to] = next_to;
            }
            ToyAction::Attach { from, to } => {
                if id != from || to >= self.degrees.len() || to == from {
                    return Err("invalid attachment".into());
                }
                self.degrees[to] = self.degrees[to].checked_add(1).ok_or("degree overflow")?;
            }
        }
        Ok(())
    }
}
pub struct RingPolicy;
impl Policy<Society> for RingPolicy {
    fn propose(&self, actor: AgentId, observed: &Observation) -> Result<ToyAction, String> {
        let from = usize::try_from(actor.0).map_err(|e| e.to_string())?;
        if from >= observed.balances.len() || observed.balances.len() < 2 {
            return Err("invalid ring actor".into());
        }
        Ok(ToyAction::Transfer {
            from,
            to: (from + 1) % observed.balances.len(),
            amount: u64::from(observed.balances[from] > 0),
        })
    }
}
pub fn wealth_total(world: &Society) -> Result<u64, String> {
    world.balances.iter().try_fold(0u64, |sum, v| {
        sum.checked_add(*v).ok_or("wealth overflow".into())
    })
}

/// A bounded preferential attachment draw with weights degree+1.
pub fn preferential_target(degrees: &[u64], rng: &mut Streams) -> Result<usize, String> {
    let sum = degrees.iter().try_fold(0u64, |sum, value| {
        sum.checked_add(value.checked_add(1).ok_or("degree overflow")?)
            .ok_or("weight overflow".into())
    })?;
    if sum == 0 {
        return Err("empty degree list".into());
    }
    let draw = rng.uniform_index(
        "attachment",
        usize::try_from(sum).map_err(|e| e.to_string())?,
    )? as u64;
    let mut cumulative = 0u64;
    for (index, degree) in degrees.iter().enumerate() {
        cumulative += degree + 1; // safe because the full sum was checked
        if draw < cumulative {
            return Ok(index);
        }
    }
    Err("unreachable weighted selection".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        experiment::{Rules, Trial},
        research::{resume_fork, verify_fork},
        runtime::ScriptedBackend,
        scheduler::Scheduling,
    };
    use std::{fs, sync::Arc};

    /// Independent reference: explicitly mutate a plain vector without World,
    /// Policy, Trial, or Event handling. This catches mistakes in the engine.
    fn independent_ring(mut balances: Vec<u64>, steps: usize) -> Vec<u64> {
        for n in 0..steps {
            let from = n % balances.len();
            let to = (from + 1) % balances.len();
            if balances[from] > 0 {
                balances[from] -= 1;
                balances[to] += 1;
            }
        }
        balances
    }

    #[tokio::test]
    async fn ring_is_symmetric_and_conserves_against_independent_reference() {
        let dir = std::env::temp_dir().join(format!("hiyal-r2-ring-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let expected = independent_ring(vec![10; 4], 160);
        let mut trial = Trial::create(
            &dir,
            "ring",
            83,
            "toy.ring",
            Society {
                balances: vec![10; 4],
                degrees: vec![0; 4],
            },
            Rules(RingPolicy),
            Arc::new(ScriptedBackend),
            (0..4).map(AgentId).collect(),
            Scheduling::RoundRobin,
        )
        .unwrap();
        assert_eq!(trial.run_attempts(160).await.unwrap(), 160);
        assert_eq!(trial.store.world().balances, expected);
        assert_eq!(trial.store.world().balances, vec![10; 4]);
        assert_eq!(wealth_total(trial.store.world()).unwrap(), 40);
        trial.checkpoint().unwrap();
        drop(trial);
        let restored = Trial::resume(
            &dir,
            "toy.ring",
            Rules(RingPolicy),
            Arc::new(ScriptedBackend),
            true,
        )
        .unwrap();
        assert_eq!(restored.store.world().balances, expected);
        drop(restored);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn preferential_attachment_matches_independent_expanded_urn() {
        let mut rng = Streams::new("preferential".into(), 120);
        let mut reference = rng.clone();
        let mut degrees = vec![0u64; 5];
        let mut other = degrees.clone();
        let mut selected = vec![0u64; 5];
        for _ in 0..2000 {
            let chosen = preferential_target(&degrees, &mut rng).unwrap();
            let mut urn = Vec::new();
            for (index, degree) in other.iter().enumerate() {
                urn.extend(std::iter::repeat_n(index, (*degree + 1) as usize));
            }
            let independent = urn[reference.uniform_index("attachment", urn.len()).unwrap()];
            assert_eq!(chosen, independent);
            selected[chosen] += 1;
            degrees[chosen] += 1;
            other[independent] += 1;
        }
        assert_eq!(degrees, other);
        assert_eq!(selected.iter().sum::<u64>(), 2000);
    }

    #[test]
    fn unbiased_uniform_sampling_has_reasonable_seeded_frequencies() {
        let mut streams = Streams::new("unbiased".into(), 122);
        let mut counts = [0u64; 5];
        for _ in 0..20_000 {
            counts[streams.uniform_index("uniform", 5).unwrap()] += 1;
        }
        for count in counts {
            assert!((count as i64 - 4000).abs() < 300);
        }
    }

    #[tokio::test]
    async fn paired_shock_has_independent_history_and_auditable_ancestry() {
        let base = std::env::temp_dir().join(format!("hiyal-r2-forks-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        let parent = base.join("parent");
        {
            let mut trial = Trial::create(
                &parent,
                "pre-shock",
                51,
                "toy.ring",
                Society {
                    balances: vec![20, 20, 20],
                    degrees: vec![0; 3],
                },
                Rules(RingPolicy),
                Arc::new(ScriptedBackend),
                (0..3).map(AgentId).collect(),
                Scheduling::Random,
            )
            .unwrap();
            trial.run_attempts(15).await.unwrap();
            trial.checkpoint().unwrap();
        }
        let root = base.join("branches");
        let (mut control, mut treatment) = crate::research::paired_forks(
            &parent,
            &root,
            "redistribution",
            serde_json::json!({"amount":3}),
            || Rules(RingPolicy),
            |world: &mut Society| {
                world.balances[0] -= 3;
                world.balances[1] += 3;
                Ok(())
            },
        )
        .unwrap();
        assert_ne!(
            control.store.genesis().trial_id,
            treatment.store.genesis().trial_id
        );
        assert_eq!(
            control.store.genesis().initial_streams.seed_domain,
            treatment.store.genesis().initial_streams.seed_domain
        );
        assert_eq!(wealth_total(control.store.world()).unwrap(), 60);
        assert_eq!(wealth_total(treatment.store.world()).unwrap(), 60);
        assert_ne!(
            control.store.world().balances,
            treatment.store.world().balances
        );
        let parent_head = verify_fork::<Society>(root.join("control"))
            .unwrap()
            .parent_head;
        assert_eq!(
            parent_head,
            verify_fork::<Society>(root.join("treatment"))
                .unwrap()
                .parent_head
        );
        control.run_attempts(12).await.unwrap();
        treatment.run_attempts(12).await.unwrap();
        control.checkpoint().unwrap();
        treatment.checkpoint().unwrap();
        let control_id = control.store.chain_head().to_owned();
        let treatment_id = treatment.store.chain_head().to_owned();
        drop(control);
        drop(treatment);
        let c = resume_fork::<Society, _>(
            &root.join("control"),
            Rules(RingPolicy),
            Arc::new(ScriptedBackend),
            true,
        )
        .unwrap();
        let t = resume_fork::<Society, _>(
            &root.join("treatment"),
            Rules(RingPolicy),
            Arc::new(ScriptedBackend),
            true,
        )
        .unwrap();
        assert_eq!(c.store.chain_head(), control_id);
        assert_eq!(t.store.chain_head(), treatment_id);
        drop(c);
        drop(t);
        fs::write(root.join("treatment/ancestry/events.jsonl"), b"corrupt\n").unwrap();
        assert!(verify_fork::<Society>(root.join("treatment")).is_err());
        fs::remove_dir_all(base).unwrap();
    }
}
