#!/usr/bin/env python3
"""Offline identity fixture verifier independent of the Rust implementation.

Uses only Python's standard library to check the *reference format*. It is a
golden-fixture tooling script; the Rust engine does not depend on Python.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
fixture = json.loads((ROOT / "fixtures/golden/scripted-two-agent-v1.json").read_text())


def derive(domain: str, parts: list[str]) -> str:
    digest = hashlib.sha256(b"hiyal:id:v1")
    for item in [domain, *parts]:
        data = item.encode("utf-8")
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


trial = derive("trial", [fixture["trial_name"], str(fixture["seed"])])
assert trial == fixture["expected_trial_id"]
assert len(fixture["expected_event_ids"]) == fixture["committed_steps"]
for step, (actor, expected) in enumerate(
    zip(fixture["expected_actor_order"], fixture["expected_event_ids"], strict=True)
):
    assert derive("event", [trial, str(step), str(actor)]) == expected
assert derive("x", ["ab", "c"]) != derive("x", ["a", "bc"])
print("R0 identity reference fixture: PASS")
