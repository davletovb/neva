import json

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


def test_deserialization_trims_oversized_persisted_history():
    state = ConversationState("agent")
    for number in range(4):
        state.record_turn("user", str(number))
    payload = state.to_dict()
    payload["max_turns"] = 2
    restored = ConversationState.from_dict(payload)
    assert [turn.message for turn in restored.turns] == ["2", "3"]


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


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "32"])
def test_invalid_turn_byte_limit_rejected(limit):
    with pytest.raises(ValueError, match="max_turn_bytes"):
        ConversationState("agent", max_turn_bytes=limit)


def test_turn_byte_limit_preserves_default_and_truncates_opt_in_history():
    unlimited = ConversationState("agent")
    bounded = ConversationState("agent", max_turn_bytes=20)
    message = "abcdefghijklmnopqrstuvwxyz"

    unlimited.record_turn("user", message)
    bounded.record_turn("user", message)

    assert unlimited.turns[0].message == message
    assert bounded.turns[0].message.endswith("...[truncated]")
    assert len(bounded.turns[0].message.encode("utf-8")) <= 20


def test_turn_byte_limit_preserves_exact_fit_and_truncates_one_byte_over():
    exact = ConversationState("agent", max_turn_bytes=20)
    over = ConversationState("agent", max_turn_bytes=20)

    exact.record_turn("user", "x" * 20)
    over.record_turn("user", "x" * 21)

    assert exact.turns[0].message == "x" * 20
    assert over.turns[0].message.endswith("...[truncated]")
    assert len(over.turns[0].message.encode("utf-8")) <= 20


def test_turn_byte_limit_is_utf8_safe():
    state = ConversationState("agent", max_turn_bytes=18)
    state.record_turn("user", "🙂🙂🙂🙂🙂🙂🙂🙂")

    stored = state.turns[0].message

    assert stored.endswith("...[truncated]")
    assert len(stored.encode("utf-8")) <= 18


def test_tiny_turn_byte_limit_still_stays_within_bound():
    state = ConversationState("agent", max_turn_bytes=3)
    state.record_turn("user", "very long message")

    assert state.turns[0].message == "..."
    assert len(state.turns[0].message.encode("utf-8")) == 3


def test_serialization_preserves_turn_byte_limit_and_legacy_payloads():
    state = ConversationState("agent", max_turn_bytes=20)
    state.record_turn("user", "abcdefghijklmnopqrstuvwxyz")

    restored = ConversationState.from_dict(state.to_dict())

    assert restored.max_turn_bytes == 20
    assert restored.turns[0].message == state.turns[0].message
    legacy = ConversationState.from_dict({"agent_name": "old", "turns": []})
    assert legacy.max_turn_bytes is None


def test_unlimited_constructor_preserves_supplied_turn_identity():
    original_turn = ConversationTurn("user", "unchanged")
    state = ConversationState("agent", turns=[original_turn])

    assert state.turns[0] is original_turn


def test_constructor_bounds_supplied_turns_without_mutating_source():
    original_turn = ConversationTurn("user", "abcdefghijklmnopqrstuvwxyz")
    original = [original_turn]

    state = ConversationState("agent", turns=original, max_turn_bytes=20)

    assert original_turn.message == "abcdefghijklmnopqrstuvwxyz"
    assert state.turns[0].message.endswith("...[truncated]")
    assert state.turns[0] is not original_turn


def test_turn_and_byte_limits_compose():
    state = ConversationState("agent", max_turns=2, max_turn_bytes=16)
    for number in range(4):
        state.record_turn("user", f"{number}-" + ("x" * 40))

    assert len(state.turns) == 2
    assert all(len(turn.message.encode("utf-8")) <= 16 for turn in state.turns)
    assert all(turn.message.endswith("...[truncated]") for turn in state.turns)


def test_agent_returns_full_response_while_storing_bounded_history():
    from neva.agents import TransformerAgent

    full_response = "r" * 100
    state = ConversationState("agent", max_turn_bytes=24)
    agent = TransformerAgent(name="agent", llm_backend=lambda _: full_response)
    agent.set_conversation_state(state)

    response = agent.receive("hello", sender="user")

    assert response == full_response
    assert state.turns[-1].message != full_response
    assert state.turns[-1].message.endswith("...[truncated]")
    assert len(state.turns[-1].message.encode("utf-8")) <= 24


def test_environment_checkpoint_keeps_turn_byte_policy():
    from neva.agents import TransformerAgent
    from neva.environments import Environment
    from neva.schedulers import RoundRobinScheduler
    from neva.utils.state_management import SimulationSnapshot

    env = Environment(RoundRobinScheduler())
    agent = TransformerAgent(name="agent", llm_backend=lambda _: "r" * 100)
    agent.set_conversation_state(ConversationState(agent.name, max_turns=4, max_turn_bytes=24))
    env.register_agent(agent)

    env.run(2)
    snapshot = SimulationSnapshot.from_json(env.snapshot().to_json())
    env.run(1)
    env.restore(snapshot)

    assert agent.conversation_state.max_turns == 4
    assert agent.conversation_state.max_turn_bytes == 24
    assert all(len(turn.message.encode("utf-8")) <= 24 for turn in agent.conversation_state.turns)


def test_surrogate_replacement_is_explicit_and_per_code_point():
    lone = json.loads('"\\ud800"')
    pair = json.loads('"\\ud800\\udc00"')

    lone_state = ConversationState("agent", max_turn_bytes=100)
    pair_state = ConversationState("agent", max_turn_bytes=100)
    lone_state.record_turn("user", lone)
    pair_state.record_turn("user", pair)

    assert lone_state.turns[0].message == "?"
    assert pair_state.turns[0].message == "??"


def test_bounded_agent_history_normalizes_unpaired_surrogates_without_failing():
    from neva.agents import TransformerAgent

    surrogate = json.loads('"\\ud800"')
    full_response = surrogate + ("r" * 40)
    state = ConversationState("agent", max_turn_bytes=24)
    agent = TransformerAgent(name="agent", llm_backend=lambda _: full_response)
    agent.set_conversation_state(state)

    response = agent.receive("hello", sender="user")
    stored = state.turns[-1].message

    assert response == full_response
    assert surrogate not in stored
    assert stored.endswith("...[truncated]")
    assert len(stored.encode("utf-8")) <= 24
