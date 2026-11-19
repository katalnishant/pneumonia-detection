"""Single-image inference with everything the study learned on validation data:
calibrated probabilities (temperature), tuned threshold, MC-dropout uncertainty and the
referral rule. Used by the Streamlit app and as a CLI.

    python -m src.inference path/to/xray.jpeg --task binary
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import keras
import numpy as np

from src import config
from src.analyze import apply_temperature
from src.data import load_gray
from src.predict import as_matrix, split_head


@dataclass
class Bundle:
    task: str
    name: str
    model: keras.Model
    temperature: float
    threshold: float | None
    referral_cutoff: float
    feat: keras.Model
    head: keras.Model

    @property
    def img_size(self):
        return self.model.input_shape[1]

    @property
    def classes(self):
        return config.CLASS_NAMES[self.task]


def load_bundle(task: str, name: str | None = None) -> Bundle:
    res = json.loads((config.task_dir(task) / "results.json").read_text())
    name = name or res["best_single_model"]
    m = next(r for r in res["models"] if r["name"] == name)
    model = keras.models.load_model(config.models_dir(task) / f"{name}.keras")
    feat, head = split_head(model)
    return Bundle(task, name, model, m["temperature"], m["threshold"],
                  m["referral"]["uncertainty_cutoff"], feat, head)


def to_input(img, img_size: int) -> np.ndarray:
    """PIL image or path -> (1, S, S, 3) float32 in [0,255] (same as training)."""
    g = load_gray(img, img_size).astype("float32")
    return np.repeat(g[None, ..., None], 3, axis=-1)


def predict(b: Bundle, x: np.ndarray, mc_samples: int = config.MC_DROPOUT_SAMPLES) -> dict:
    f = b.feat(x, training=False)
    samples = np.stack([as_matrix(np.asarray(b.head(f, training=True)), b.task)
                        for _ in range(mc_samples)])[:, 0]          # (T, C)
    probs = apply_temperature(samples.mean(0, keepdims=True), b.temperature)[0]
    if b.task == "binary":
        pred = int(probs[1] >= b.threshold)
    else:
        pred = int(probs.argmax())
    unc = float(samples[:, pred].std())
    return {"label": b.classes[pred], "pred": pred,
            "probs": {c: float(p) for c, p in zip(b.classes, probs)},
            "uncertainty": unc, "refer": unc > b.referral_cutoff}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--task", choices=list(config.CLASS_NAMES), default="binary")
    p.add_argument("--model", default=None, help="default: best model of the study")
    a = p.parse_args()
    b = load_bundle(a.task, a.model)
    r = predict(b, to_input(a.image, b.img_size))
    print(json.dumps({"model": b.name, **r}, indent=2))
