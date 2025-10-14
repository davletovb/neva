import sys

import pytest

from neva.memory import FaissVectorStoreMemory, MemoryConfigurationError


@pytest.fixture(scope="module")
def _skip_if_faiss_missing():
    pytest.importorskip("faiss")
    pytest.importorskip("numpy")


def _toy_embedder(text: str):
    # Simple deterministic embedding based on character codes.
    import numpy as np

    vector = np.zeros(16, dtype="float32")
    for index, byte in enumerate(text.encode("utf-8")):
        vector[index % 16] += float(byte) / 255.0
    return vector


@pytest.mark.usefixtures("_skip_if_faiss_missing")
def test_faiss_memory_recalls_similar_messages():
    memory = FaissVectorStoreMemory(_toy_embedder, top_k=2)
    memory.remember("Alice", "Discuss project timeline")
    memory.remember("Bob", "Review architecture draft")
    memory.remember("Alice", "Finalize project timeline")

    summary = memory.recall(query="project timeline")
    assert "Finalize project timeline" in summary
    assert "Discuss project timeline" in summary
    assert "architecture" not in summary


def test_requires_faiss_dependency(monkeypatch):
    if "faiss" in sys.modules:
        pytest.skip("faiss installed; cannot test missing dependency")

    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "faiss":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(MemoryConfigurationError):
        FaissVectorStoreMemory(_toy_embedder)
