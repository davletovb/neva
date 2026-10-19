//! Standalone application crate: custom world and policy without changing Hiyal.

use hiyal::core::{AgentId, Engine, Policy, TrialId, World};

#[derive(Clone)]
struct Society {
    total_actions: u64,
}

impl World for Society {
    type Action = u64;
    type Observation = u64;

    fn observe(&self, _actor: AgentId) -> Result<Self::Observation, String> {
        Ok(self.total_actions)
    }

    fn apply(&mut self, _actor: AgentId, action: Self::Action) -> Result<(), String> {
        self.total_actions = self.total_actions.checked_add(action).ok_or("overflow")?;
        Ok(())
    }
}

struct Cooperate;
impl Policy<Society> for Cooperate {
    fn propose(&self, _actor: AgentId, _observation: &u64) -> Result<u64, String> {
        Ok(1)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut simulation = Engine::new(
        TrialId::new("cooperation-demo", 42),
        Society { total_actions: 0 },
        Cooperate,
        [AgentId(0), AgentId(1), AgentId(2)],
    )?;
    for _ in 0..6 {
        simulation.step()?;
    }
    assert_eq!(simulation.world.total_actions, 6);
    simulation.write_jsonl(std::io::stdout())?;
    Ok(())
}
