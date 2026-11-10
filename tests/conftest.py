import numpy as np
import pytest

from dgrag.store import FaissManager, SQLiteStore

DIM = 8


@pytest.fixture
def store(tmp_path):
    s = SQLiteStore(tmp_path / "test.sqlite")
    yield s
    s.close()


@pytest.fixture
def faiss_mgr(store, tmp_path):
    return FaissManager(store, tmp_path / "faiss", DIM)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def rand_vec(rng, n=None):
    shape = (DIM,) if n is None else (n, DIM)
    v = rng.standard_normal(shape).astype(np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)
