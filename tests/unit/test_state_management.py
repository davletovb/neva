from pathlib import Path

from neva.utils.state_management import (
    ConversationState,
    create_snapshot,
    load_snapshot,
    save_snapshot,
)


def test_snapshot_roundtrip(tmp_path: Path) -> None:
    state = ConversationState(agent_name="agent")
    state.record_turn("user", "hello")
    snapshot = create_snapshot(environment_state={"mood": "calm"}, agent_states=[state])

    path = tmp_path / "snapshot.json"
    save_snapshot(snapshot, path)

    loaded = load_snapshot(path)
    assert loaded.environment_state == {"mood": "calm"}
    assert "agent" in loaded.agent_states
    turns = loaded.agent_states["agent"]["turns"]
    assert turns[0]["message"] == "hello"
