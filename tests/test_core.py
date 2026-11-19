"""Fast unit tests for the parts most likely to silently break results.

    pytest -q
"""
import numpy as np
import pytest

from src import config
from src.analyze import apply_temperature, ece, fit_temperature, referral
from src.data import patient_id, subtype_label, to_task_labels


def test_patient_id_groups_same_person():
    assert patient_id("x/PNEUMONIA/person1000_virus_1681.jpeg") == "PNEUMONIA/person1000"
    assert patient_id("x/PNEUMONIA/person1000_bacteria_99.jpeg") == "PNEUMONIA/person1000"
    assert patient_id("x/NORMAL/IM-0115-0001.jpeg") == "NORMAL/IM-0115"
    assert patient_id("x/NORMAL/NORMAL2-IM-0329-0001.jpeg") == "NORMAL/NORMAL2-IM-0329"


def test_subtype_and_task_labels():
    subs = [subtype_label(p) for p in ["a/NORMAL/IM-1-1.jpeg", "a/PNEUMONIA/person1_bacteria_1.jpeg",
                                       "a/PNEUMONIA/person2_virus_3.jpeg"]]
    assert subs == [0, 1, 2]
    assert to_task_labels(subs, "binary").tolist() == [0, 1, 1]
    assert to_task_labels(subs, "multiclass").tolist() == [0, 1, 2]


def test_temperature_scaling_reduces_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    # 75%-accurate model that always claims 99% confidence
    pred = np.where(rng.random(2000) < 0.75, y, 1 - y)
    P = np.where(pred[:, None] == np.arange(2), 0.99, 0.01)
    t = fit_temperature(y, P)
    assert t > 1.0
    assert ece(y, apply_temperature(P, t)) < ece(y, P) / 3


def test_referral_removes_uncertain_errors():
    y = np.array([0, 1] * 50)
    P = np.eye(2)[y].astype(float)
    P[:10] = P[:10][:, ::-1]                  # first 10 predictions are wrong...
    u = np.zeros(100)
    u[:10] = 1.0                              # ...and flagged as uncertain
    r = referral(y, P, u, y, P, u, thr=0.5, rate=0.1)
    assert r["errors_caught"] == 10 and r["accuracy_kept"] == 1.0


@pytest.mark.parametrize("name", ["vgg16", "densenet121"])
def test_inmodel_preprocessing_matches_keras(name):
    keras = pytest.importorskip("keras")
    from src.models import BACKBONES, preprocess_layer
    x = np.repeat(np.random.default_rng(0).uniform(0, 255, (2, 8, 8, 1)), 3, -1).astype("float32")
    mod = keras.applications.vgg16 if name == "vgg16" else keras.applications.densenet
    ref = mod.preprocess_input(x.copy())
    ours = np.asarray(preprocess_layer(BACKBONES[name][1])(x))
    np.testing.assert_allclose(ours, ref, atol=1e-3)


def test_models_build_for_both_tasks():
    pytest.importorskip("keras")
    from src.models import build_model
    for task, n in [("binary", 1), ("multiclass", 3)]:
        for name in config.MODEL_NAMES:
            m = build_model(name, task, img_size=32, weights=None)
            assert m.output_shape[-1] == n
