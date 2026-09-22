import io
from pathlib import Path

import pytest

import neva.utils.state_management as state_management
from neva.utils.state_management import create_snapshot, load_snapshot, save_snapshot


def test_exact_byte_limit_roundtrip_and_one_byte_overflow(tmp_path):
    snapshot = create_snapshot(environment_state={"message": "你好 🌍"})
    raw = snapshot.to_json().encode("utf-8")
    path = tmp_path / "snapshot.json"
    save_snapshot(snapshot, path, max_bytes=len(raw))
    assert path.read_bytes() == raw
    assert load_snapshot(path, max_bytes=len(raw)) == snapshot
    with pytest.raises(ValueError, match="exceeds max_bytes"):
        load_snapshot(path, max_bytes=len(raw) - 1)
    with pytest.raises(ValueError, match="exceeds max_bytes"):
        save_snapshot(snapshot, path, max_bytes=len(raw) - 1)
    assert path.read_bytes() == raw


def test_utf8_file_limit_counts_bytes_not_characters(tmp_path):
    raw = create_snapshot(environment_state={"message": "你好 🌍"}).to_json()
    # Files from external JSON writers need not escape non-ASCII characters.
    import json

    raw = json.dumps(json.loads(raw), ensure_ascii=False)
    path = tmp_path / "snapshot.json"
    path.write_bytes(raw.encode("utf-8"))
    assert len(raw.encode("utf-8")) > len(raw)
    with pytest.raises(ValueError, match="exceeds max_bytes"):
        load_snapshot(path, max_bytes=len(raw))
    assert load_snapshot(path, max_bytes=len(raw.encode("utf-8"))).environment_state == {
        "message": "你好 🌍"
    }


def test_limited_load_reads_only_limit_plus_one(monkeypatch):
    sizes = []

    class RecordingReader(io.BytesIO):
        def read(self, size=-1):
            sizes.append(size)
            return super().read(size)

    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: RecordingReader(b"x" * 100))
    with pytest.raises(ValueError, match="exceeds max_bytes"):
        load_snapshot(Path("checkpoint.json"), max_bytes=10)
    assert sizes == [11]


def test_large_limit_does_not_preallocate_ceiling(monkeypatch):
    reads = []

    class RecordingReader(io.BytesIO):
        def read(self, size=-1):
            reads.append(size)
            return super().read(size)

    minimal_payload = (
        b'{"created_at": "2026-09-17T00:00:00", '
        b'"environment_state": {}, "agent_states": {}, "version": 1}'
    )
    monkeypatch.setattr(
        Path,
        "open",
        lambda *args, **kwargs: RecordingReader(minimal_payload),
    )
    snapshot = load_snapshot(Path("checkpoint.json"), max_bytes=2**40)
    assert snapshot.version == 1
    assert max(reads) <= 65536
    assert sum(reads) <= 131072


def test_large_limit_loads_multi_chunk_file(tmp_path):
    payload = b'{"created_at": "2026-09-17T00:00:00", "environment_state": {"k": "'
    payload = payload + b"v" * 150_000 + b'"}, "agent_states": {}, "version": 1}'
    path = tmp_path / "snapshot.json"
    path.write_bytes(payload)
    assert len(payload) > 65536
    snapshot = load_snapshot(path, max_bytes=200_000)
    assert snapshot.environment_state["k"] == "v" * 150_000
    with pytest.raises(ValueError, match="exceeds max_bytes"):
        load_snapshot(path, max_bytes=len(payload) - 1)


def test_oversized_save_does_not_create_file(tmp_path):
    path = tmp_path / "snapshot.json"
    with pytest.raises(ValueError, match="exceeds max_bytes"):
        save_snapshot(create_snapshot(), path, max_bytes=1)
    assert not path.exists()


def test_oversized_save_preserves_existing_file(tmp_path):
    snapshot = create_snapshot(environment_state={"message": "large" * 100})
    path = tmp_path / "snapshot.json"
    path.write_text("existing checkpoint", encoding="utf-8")

    with pytest.raises(ValueError, match="exceeds max_bytes"):
        save_snapshot(snapshot, path, max_bytes=10)

    assert path.read_text(encoding="utf-8") == "existing checkpoint"


@pytest.mark.parametrize("limit", [0, -1, True, False, 1.5, "10"])
@pytest.mark.parametrize("operation", ["save", "load"])
def test_invalid_limits_rejected_before_file_access(tmp_path, limit, operation):
    path = tmp_path / "absent" / "snapshot.json"
    with pytest.raises(ValueError, match="max_bytes must be a positive integer or None"):
        if operation == "save":
            save_snapshot(create_snapshot(), path, max_bytes=limit)
        else:
            load_snapshot(path, max_bytes=limit)
    assert not path.exists()


def test_oversized_load_rejected_before_decode_or_parse(tmp_path):
    path = tmp_path / "snapshot.json"
    path.write_bytes(b"\xff" * 100)
    with pytest.raises(ValueError, match="exceeds max_bytes"):
        load_snapshot(path, max_bytes=10)


def test_save_streams_without_calling_to_json(tmp_path, monkeypatch):
    snapshot = create_snapshot(environment_state={"items": list(range(1000))})
    path = tmp_path / "snapshot.json"

    def fail_to_json():
        raise AssertionError("save_snapshot should not materialise to_json()")

    monkeypatch.setattr(snapshot, "to_json", fail_to_json)

    save_snapshot(snapshot, path)

    loaded = load_snapshot(path)
    assert loaded.environment_state == snapshot.environment_state


def test_save_streams_to_sibling_temp_file_and_cleans_it(tmp_path, monkeypatch):
    calls = []
    writes = []
    real_factory = state_management.tempfile.NamedTemporaryFile

    class RecordingFile:
        def __init__(self, inner):
            self._inner = inner
            self.name = inner.name

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return self._inner.__exit__(exc_type, exc, tb)

        def write(self, data):
            writes.append(len(data))
            return self._inner.write(data)

        def flush(self):
            return self._inner.flush()

        def fileno(self):
            return self._inner.fileno()

    def recording_temp_file(**kwargs):
        calls.append(kwargs)
        return RecordingFile(real_factory(**kwargs))

    monkeypatch.setattr(
        state_management.tempfile,
        "NamedTemporaryFile",
        recording_temp_file,
    )

    snapshot = create_snapshot(
        environment_state={"items": [{"value": index} for index in range(5000)]}
    )
    path = tmp_path / "snapshot.json"
    expected = snapshot.to_json().encode("utf-8")

    save_snapshot(snapshot, path)

    assert len(calls) == 1
    assert calls[0]["dir"] == str(tmp_path)
    assert calls[0]["delete"] is False
    assert calls[0]["prefix"] == ".snapshot.json."
    assert calls[0]["suffix"] == ".tmp"
    assert len(writes) > 1
    assert max(writes) < len(expected)
    assert path.read_bytes() == expected
    assert list(tmp_path.glob(".snapshot.json.*.tmp")) == []


def test_serialization_failure_preserves_existing_file(tmp_path):
    class Unsupported:
        pass

    path = tmp_path / "snapshot.json"
    path.write_text("existing checkpoint", encoding="utf-8")
    snapshot = create_snapshot(environment_state={"bad": Unsupported()})

    with pytest.raises(TypeError, match="not JSON serialisable"):
        save_snapshot(snapshot, path)

    assert path.read_text(encoding="utf-8") == "existing checkpoint"


def test_staging_write_failure_preserves_existing_file_and_cleans_temp(tmp_path, monkeypatch):
    real_factory = state_management.tempfile.NamedTemporaryFile
    writes = 0

    class FailingFile:
        def __init__(self, inner):
            self._inner = inner
            self.name = inner.name

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return self._inner.__exit__(exc_type, exc, tb)

        def write(self, data):
            nonlocal writes
            writes += 1
            if writes > 2:
                raise OSError("simulated disk failure")
            return self._inner.write(data)

        def flush(self):
            return self._inner.flush()

        def fileno(self):
            return self._inner.fileno()

    monkeypatch.setattr(
        state_management.tempfile,
        "NamedTemporaryFile",
        lambda **kwargs: FailingFile(real_factory(**kwargs)),
    )

    path = tmp_path / "snapshot.json"
    path.write_text("precious checkpoint", encoding="utf-8")
    snapshot = create_snapshot(environment_state={"message": "x" * 10_000})

    with pytest.raises(OSError, match="simulated disk failure"):
        save_snapshot(snapshot, path)

    assert path.read_text(encoding="utf-8") == "precious checkpoint"
    assert list(tmp_path.glob(".snapshot.json.*.tmp")) == []


def test_replace_failure_preserves_existing_file_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "snapshot.json"
    path.write_text("precious checkpoint", encoding="utf-8")
    snapshot = create_snapshot(environment_state={"message": "replacement"})

    def fail_replace(source, destination):
        assert Path(source).parent == tmp_path
        assert Path(destination) == path
        raise OSError("simulated replace failure")

    monkeypatch.setattr(state_management.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        save_snapshot(snapshot, path)

    assert path.read_text(encoding="utf-8") == "precious checkpoint"
    assert list(tmp_path.glob(".snapshot.json.*.tmp")) == []


def test_serialization_error_precedes_size_error(tmp_path):
    class Unsupported:
        pass

    snapshot = create_snapshot(environment_state={"large": "x" * 10_000, "bad": Unsupported()})

    with pytest.raises(TypeError, match="not JSON serialisable"):
        save_snapshot(snapshot, tmp_path / "snapshot.json", max_bytes=10)
