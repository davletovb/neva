import pytest

from neva.utils.state_management import ConversationState, ConversationTurn


def test_retention_keeps_latest_messages_without_changing_default():
    bounded = ConversationState("agent", max_turns=3)
    unlimited = ConversationState("agent")
    for number in range(10):
        for state in (bounded, unlimited):
            state.record_turn("user", str(number))
    assert [turn.message for turn in bounded.turns] == ["7", "8", "9"]
    assert len(unlimited.turns) == 10


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "3"])
def test_invalid_retention_limit_rejected(limit):
    with pytest.raises(ValueError, match="max_turns"):
        ConversationState("agent", max_turns=limit)


def test_serialization_preserves_limit_and_legacy_payloads():
    state = ConversationState("agent", max_turns=1)
    state.record_turn("user", "hello")
    restored = ConversationState.from_dict(state.to_dict())
    assert restored.max_turns == 1
    assert restored.turns[0].timestamp == state.turns[0].timestamp
    restored.record_turn("agent", "reply")
    assert [t.message for t in restored.turns] == ["reply"]
    legacy = ConversationState.from_dict({"agent_name": "old", "turns": []})
    assert legacy.max_turns is None


def test_environment_checkpoint_keeps_retention_policy():
    from neva.agents import TransformerAgent
    from neva.environments import Environment
    from neva.schedulers import RoundRobinScheduler
    from neva.utils.state_management import SimulationSnapshot

    env = Environment(RoundRobinScheduler())
    agent = TransformerAgent(name="agent", llm_backend=lambda _: "reply")
    agent.set_conversation_state(ConversationState(agent.name, max_turns=2))
    env.register_agent(agent)
    env.run(3)
    snapshot = SimulationSnapshot.from_json(env.snapshot().to_json())
    saved = agent.conversation_state.to_dict()
    env.run(2)
    env.restore(snapshot)
    assert agent.conversation_state.max_turns == 2
    assert agent.conversation_state.to_dict() == saved
    env.run(3)
    assert len(agent.conversation_state.turns) == 2
    assert agent.conversation_state.to_dict() != saved


def test_constructor_trims_a_copy_of_supplied_history():
    original = [ConversationTurn("user", str(i)) for i in range(4)]
    state = ConversationState("agent", turns=original, max_turns=2)
    assert [turn.message for turn in state.turns] == ["2", "3"]
    assert len(original) == 4
    state.record_turn("agent", "reply")
    assert len(original) == 4
