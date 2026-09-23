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


def _semantic_embedder(text: str):
    import numpy as np

    lowered = text.lower()
    if "alpha" in lowered:
        return np.asarray([1.0, 0.0, 0.0], dtype="float32")
    if "beta" in lowered:
        return np.asarray([0.0, 1.0, 0.0], dtype="float32")
    return np.asarray([0.0, 0.0, 1.0], dtype="float32")


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


@pytest.mark.usefixtures("_skip_if_faiss_missing")
def test_faiss_search_keeps_nearest_first_l2_order():
    memory = FaissVectorStoreMemory(_semantic_embedder, top_k=3)
    memory.remember("A", "alpha exact")
    memory.remember("B", "beta orthogonal")
    memory.remember("C", "gamma orthogonal")

    lines = memory.recall(query="alpha query", limit=3).splitlines()

    assert lines[0] == "A: alpha exact"
    assert set(lines[1:]) == {"B: beta orthogonal", "C: gamma orthogonal"}


@pytest.mark.usefixtures("_skip_if_faiss_missing")
@pytest.mark.parametrize(
    "embedding",
    [
        [[1.0, 2.0]],
        [[1.0], [2.0]],
    ],
)
def test_faiss_rejects_non_1d_embeddings(embedding):
    memory = FaissVectorStoreMemory(lambda _: embedding)

    with pytest.raises(MemoryConfigurationError, match="one-dimensional"):
        memory.remember("A", "bad shape")


@pytest.mark.usefixtures("_skip_if_faiss_missing")
def test_faiss_normalization_does_not_mutate_embedder_array():
    import numpy as np

    embedding = np.asarray([3.0, 4.0], dtype="float32")
    original = embedding.copy()
    memory = FaissVectorStoreMemory(lambda _: embedding)

    memory.remember("A", "owned input")

    assert np.array_equal(embedding, original)


@pytest.mark.usefixtures("_skip_if_faiss_missing")
def test_faiss_accepts_read_only_float32_embeddings():
    import numpy as np

    embedding = np.asarray([3.0, 4.0], dtype="float32")
    embedding.flags.writeable = False
    memory = FaissVectorStoreMemory(lambda _: embedding)

    memory.remember("A", "read only")

    assert memory.recall(query="anything") == "A: read only"


@pytest.mark.usefixtures("_skip_if_faiss_missing")
def test_faiss_still_accepts_generator_embeddings():
    memory = FaissVectorStoreMemory(lambda _: (value for value in (1.0, 2.0)))
    memory.remember("A", "generator")

    assert memory.recall(query="anything") == "A: generator"


@pytest.mark.usefixtures("_skip_if_faiss_missing")
def test_faiss_recent_recall_clear_and_reuse():
    memory = FaissVectorStoreMemory(_semantic_embedder, top_k=2)
    assert memory.recall() == ""

    memory.remember("A", "alpha one")
    memory.remember("B", "beta two")
    memory.remember("C", "gamma three")
    assert memory.recall() == "B: beta two\nC: gamma three"

    memory.clear()
    assert memory.recall(query="alpha") == ""
    assert memory._id_counter == 0
    assert memory._records == {}
    assert memory._order == []

    memory.remember("A", "alpha after clear")
    assert memory.recall(query="alpha") == "A: alpha after clear"


def test_faiss_top_k_must_be_positive_before_optional_import():
    with pytest.raises(MemoryConfigurationError, match="top_k"):
        FaissVectorStoreMemory(_toy_embedder, top_k=0)


def test_requires_faiss_dependency(monkeypatch):
    original_import = __import__
    monkeypatch.delitem(sys.modules, "faiss", raising=False)

    def fake_import(name, *args, **kwargs):
        if name == "faiss":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(MemoryConfigurationError, match="faiss"):
        FaissVectorStoreMemory(_toy_embedder)


@pytest.mark.usefixtures("_skip_if_faiss_missing")
def test_requires_numpy_alongside_faiss(monkeypatch):
    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "numpy":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(MemoryConfigurationError, match="NumPy"):
        FaissVectorStoreMemory(_toy_embedder)
