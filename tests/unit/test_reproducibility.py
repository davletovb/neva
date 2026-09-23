import json
import random

import pytest

from neva.agents import GPTAgent, TransformerAgent
from neva.agents.base import AIAgent
from neva.environments import BasicEnvironment, Environment
from neva.schedulers import CompositeScheduler, RandomScheduler
from neva.utils.caching import LLMCache
from neva.utils.exceptions import RecordedReplayError, ReplayMismatchError, ReproducibilityError
from neva.utils.reproducibility import (
    ReplayRecord,
    ReplayTape,
    RunManifest,
    create_run_manifest,
    prepare_reproducible_run,
    seed_everything,
)


class SeedAwareAgent(AIAgent):
    def __init__(self, name):
        super().__init__(name=name)
        self.seed_seen = None

    def set_seed(self, seed):
        self.seed_seen = seed

    def respond(self, message):
        return f"{self.name}:{message}"


def _deterministic_backend(prompt):
    return f"reply:{len(prompt)}:{prompt[:24]}"


def _build_random_env(backend=_deterministic_backend):
    env = BasicEnvironment("lab", "deterministic replay", RandomScheduler())
    for name in ("alpha", "beta", "gamma"):
        env.register_agent(TransformerAgent(name=name, llm_backend=backend))
    return env


@pytest.mark.parametrize("bad_seed", [True, 1.5, "7", -1])
def test_seed_everything_rejects_invalid_seed(bad_seed):
    with pytest.raises(ReproducibilityError, match="seed"):
        seed_everything(bad_seed, optional_libraries=False)


def test_seed_everything_repeats_global_random_and_scheduler_sequence():
    env_one = _build_random_env()
    report_one = seed_everything(1234, environment=env_one, optional_libraries=False)
    global_one = [random.random() for _ in range(4)]
    selected_one = [env_one.scheduler.get_next_agent().name for _ in range(12)]

    env_two = _build_random_env()
    report_two = seed_everything(1234, environment=env_two, optional_libraries=False)
    global_two = [random.random() for _ in range(4)]
    selected_two = [env_two.scheduler.get_next_agent().name for _ in range(12)]

    assert global_one == global_two
    assert selected_one == selected_two
    assert report_one.scheduler_seeds == report_two.scheduler_seeds
    assert report_one.optional_libraries == {"numpy": "skipped", "torch": "skipped"}
    assert "child-processes" in report_one.python_hash_seed


def test_environment_seed_reaches_nested_composite_scheduler_and_agent_hook():
    outer = CompositeScheduler()
    inner_random = RandomScheduler()
    agent = SeedAwareAgent("seed-aware")
    outer.add(agent, group="random-group", scheduler=inner_random)
    env = Environment(outer)
    agent.set_environment(env)
    env.agents.append(agent)

    report = env.seed(77, optional_libraries=False)

    assert agent.seed_seen == report.agent_seeds["seed-aware"]
    assert "scheduler[0].group[random-group]" in report.scheduler_seeds
    assert isinstance(inner_random._rng, random.Random)


def test_manifest_captures_prompts_provider_model_generation_cache_and_dependencies():
    env = BasicEnvironment("study", "manifest", RandomScheduler())
    agent = GPTAgent(
        name="writer",
        api_key="super-secret-key",
        provider="openai",
        model="gpt-test",
        llm_backend=_deterministic_backend,
        cache=LLMCache(max_size=7),
        max_retries=2,
        retry_backoff=1.25,
        max_output_tokens=321,
        max_context_chars=4321,
        request_timeout=9.0,
    )
    env.register_agent(agent)
    report = seed_everything(9, environment=env, optional_libraries=False)

    manifest = create_run_manifest(
        env,
        seed=9,
        seed_report=report,
        prompts={"system": "Grade carefully", "scenario": "Essay task"},
        dependencies=["requests", "definitely-not-installed-neva-test-package"],
        metadata={"experiment": "baseline"},
    )
    payload = manifest.to_dict()
    agent_config = payload["agents"][0]

    assert payload["seed"] == 9
    assert payload["prompts"] == {"scenario": "Essay task", "system": "Grade carefully"}
    assert payload["scheduler"]["type"].endswith(".RandomScheduler")
    assert payload["scheduler"]["rng_state_sha256"]
    assert agent_config["provider"] == "openai"
    assert agent_config["model"] == "gpt-test"
    assert agent_config["generation"]["max_output_tokens"] == 321
    assert agent_config["generation"]["max_context_chars"] == 4321
    assert agent_config["generation"]["max_retries"] == 2
    assert agent_config["generation"]["retry_backoff"] == 1.25
    assert agent_config["generation"]["request_timeout"] == 9.0
    assert agent_config["cache"]["enabled"] is True
    assert agent_config["cache"]["max_size"] == 7
    assert payload["dependencies"]["requests"]
    assert payload["dependencies"]["definitely-not-installed-neva-test-package"] is None
    assert payload["metadata"] == {"experiment": "baseline"}
    assert "super-secret-key" not in json.dumps(payload)


def test_manifest_marks_builtin_live_provider_as_not_exactly_reproducible():
    env = BasicEnvironment("live", "provider caveat", RandomScheduler())
    agent = GPTAgent(
        name="live-agent",
        api_key="not-used",
        provider="openai",
        model="gpt-live-test",
        provider_rate=None,
        max_provider_concurrency=None,
    )
    env.register_agent(agent)

    manifest = create_run_manifest(env, seed=1, dependencies=[])

    assert manifest.reproducibility["live_providers"] == ["openai"]
    assert manifest.reproducibility["live_provider_exact_replay"] is False
    assert any(
        "not guaranteed reproducible" in note
        for note in manifest.reproducibility["notes"]
    )


def test_manifest_round_trip_and_fingerprint_ignore_creation_time(tmp_path):
    env = _build_random_env()
    first = prepare_reproducible_run(
        env,
        seed=44,
        prompts=["one", "two"],
        dependencies=["requests"],
        optional_libraries=False,
    )
    path = tmp_path / "manifest.json"
    first.save(path)
    loaded = RunManifest.load(path)

    payload = loaded.to_dict()
    payload["created_at"] = "2099-01-01T00:00:00+00:00"
    changed_time = RunManifest.from_dict(payload)

    assert loaded.to_dict() == first.to_dict()
    assert changed_time.fingerprint() == first.fingerprint()
    assert list(tmp_path.glob(".manifest.json.*.tmp")) == []


@pytest.mark.parametrize(
    "prompts",
    [
        "single string is ambiguous",
        {"ok": 1},
        ["ok", 2],
    ],
)
def test_manifest_rejects_invalid_prompt_shapes(prompts):
    env = _build_random_env()
    with pytest.raises(ReproducibilityError, match="prompt"):
        create_run_manifest(env, seed=1, prompts=prompts, dependencies=[])


def test_record_and_replay_reproduce_seeded_random_run_end_to_end(tmp_path):
    recording_env = _build_random_env()
    recording_manifest = prepare_reproducible_run(
        recording_env,
        seed=2026,
        prompts={"scenario": recording_env.context()},
        dependencies=["requests"],
        optional_libraries=False,
    )
    tape = ReplayTape.for_manifest(recording_manifest)
    recorder = tape.recording_backend(_deterministic_backend)
    for agent in recording_env.agents:
        agent.set_llm_backend(recorder)

    recorded_outputs = [recording_env.step() for _ in range(12)]
    recorded_turns = {
        agent.name: [turn.to_dict() for turn in agent.conversation_state.turns]
        for agent in recording_env.agents
    }

    manifest_path = tmp_path / "run-manifest.json"
    tape_path = tmp_path / "replay.json"
    recording_manifest.save(manifest_path)
    tape.save(tape_path)

    replay_env = _build_random_env()
    replay_manifest = prepare_reproducible_run(
        replay_env,
        seed=2026,
        prompts={"scenario": replay_env.context()},
        dependencies=["requests"],
        optional_libraries=False,
    )
    assert replay_manifest.fingerprint() == recording_manifest.fingerprint()

    loaded_tape = ReplayTape.load(tape_path)
    replay = loaded_tape.replay_backend(manifest=RunManifest.load(manifest_path))
    for agent in replay_env.agents:
        agent.set_llm_backend(replay)

    replayed_outputs = [replay_env.step() for _ in range(12)]
    replayed_turns = {
        agent.name: [turn.to_dict() for turn in agent.conversation_state.turns]
        for agent in replay_env.agents
    }
    replay.assert_consumed()

    assert replayed_outputs == recorded_outputs
    assert replayed_turns == recorded_turns
    assert len(loaded_tape.records) == 12


def test_replay_rejects_prompt_mismatch_without_advancing():
    tape = ReplayTape()
    tape.recording_backend(lambda prompt: "ok")("expected")
    replay = tape.replay_backend()

    with pytest.raises(ReplayMismatchError, match="prompt mismatch"):
        replay("changed")

    assert replay.position == 0
    assert replay("expected") == "ok"
    assert replay.position == 1


def test_replay_exhaustion_and_unconsumed_records_are_explicit():
    tape = ReplayTape()
    recorder = tape.recording_backend(lambda prompt: prompt.upper())
    recorder("one")
    recorder("two")

    replay = tape.replay_backend()
    assert replay("one") == "ONE"
    with pytest.raises(ReplayMismatchError, match="consumed 1 of 2"):
        replay.assert_consumed()
    assert replay("two") == "TWO"
    replay.assert_consumed()
    with pytest.raises(ReplayMismatchError, match="exhausted"):
        replay("three")


def test_replay_manifest_mismatch_is_rejected():
    env = _build_random_env()
    manifest = prepare_reproducible_run(
        env,
        seed=1,
        dependencies=[],
        optional_libraries=False,
    )
    tape = ReplayTape.for_manifest(manifest)

    payload = manifest.to_dict()
    payload["seed"] = 2
    different = RunManifest.from_dict(payload)

    with pytest.raises(ReplayMismatchError, match="manifest"):
        tape.replay_backend(manifest=different)


def test_recorded_backend_failure_replays_as_recorded_error():
    tape = ReplayTape()

    def fail(prompt):
        raise ValueError(f"bad:{prompt}")

    recorder = tape.recording_backend(fail)
    with pytest.raises(ValueError, match="bad:oops"):
        recorder("oops")

    record = tape.records[0]
    assert record.error_type.endswith(".ValueError")
    replay = tape.replay_backend()
    with pytest.raises(RecordedReplayError, match=r"ValueError: bad:oops"):
        replay("oops")


def test_replay_tape_detects_tampered_prompt_digest():
    record = ReplayRecord(
        prompt="original",
        prompt_sha256="0" * 64,
        response="x",
    )
    payload = {"version": 1, "manifest_fingerprint": None, "records": [record.to_dict()]}

    with pytest.raises(ReproducibilityError, match="digest"):
        ReplayTape.from_dict(payload)


def test_recording_backend_rejects_non_string_response_and_records_failure():
    tape = ReplayTape()
    recorder = tape.recording_backend(lambda prompt: 123)

    with pytest.raises(ReproducibilityError, match="response"):
        recorder("prompt")

    assert len(tape.records) == 1
    assert tape.records[0].error_type.endswith(".ReproducibilityError")
