import pytest

np = pytest.importorskip("numpy")

from perceptrome.classical_baselines import TreeSequenceBaseline


def test_tree_baseline_majority_backend_predict_shapes(tmp_path):
    x = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.array([0, 1, 1, 1], dtype=np.int64)

    model = TreeSequenceBaseline(backend="majority", random_state=7).fit(x, y)
    probs = model.predict_proba(x)
    pred = model.predict(x)

    assert model.backend_used == "majority"
    assert probs.shape == (4, 2)
    assert pred.shape == (4,)

    path = tmp_path / "tree.pkl"
    model.save(str(path))
    loaded = TreeSequenceBaseline.load(str(path))
    loaded_probs = loaded.predict_proba(x)
    assert np.allclose(loaded_probs, probs)


def test_tree_baseline_auto_backend_falls_back_to_supported_choice():
    x = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.array([0, 1, 1, 1], dtype=np.int64)

    model = TreeSequenceBaseline(backend="auto").fit(x, y)

    assert model.backend_used in {"xgboost", "sklearn", "majority"}
    assert model.predict(x).shape == (4,)
