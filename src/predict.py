"""Inference for a trained model on val + test: plain, test-time augmentation (TTA) and
Monte-Carlo dropout. Results go to artifacts/<task>/preds/<model>.npz for analysis.

    python -m src.predict --task binary --model densenet121
"""
from __future__ import annotations

import argparse

import keras
import numpy as np
import tensorflow as tf

from src import config
from src.data import load_all, to_model_input

# TTA views as crop boxes (y1, x1, y2, x2) in relative coords, each resized back to full size:
# identity, 8% centre zoom, and small shifts. All are plausible re-positionings of a patient.
TTA_BOXES = np.array([
    [0.00, 0.00, 1.00, 1.00],
    [0.04, 0.04, 0.96, 0.96],
    [0.00, 0.00, 0.94, 0.94],
    [0.06, 0.06, 1.00, 1.00],
    [0.00, 0.06, 0.94, 1.00],
], dtype="float32")


def as_matrix(p: np.ndarray, task: str) -> np.ndarray:
    """Binary sigmoid output (N,1) -> (N,2) class-probability matrix; softmax stays (N,C)."""
    p = np.asarray(p, dtype="float64")
    if task == "binary":
        p = p.reshape(-1)
        return np.stack([1 - p, p], axis=1)
    return p


def _batches(x, batch_size=64):
    return tf.data.Dataset.from_tensor_slices(x).batch(batch_size).map(to_model_input)


def predict_plain(model, x) -> np.ndarray:
    return model.predict(_batches(x), verbose=0)


def predict_tta(model, x) -> np.ndarray:
    size = model.input_shape[1:3]
    preds = []
    for box in TTA_BOXES:
        def view(xb, box=box):
            n = tf.shape(xb)[0]
            boxes = tf.tile(box[None], [n, 1])
            return tf.image.crop_and_resize(xb, boxes, tf.range(n), size)
        ds = _batches(x).map(view)
        preds.append(model.predict(ds, verbose=0))
    return np.mean(preds, axis=0)


def split_head(model: keras.Model):
    """Split a model into (feature extractor, head) at the pooling layer (gap / flatten)."""
    names = [l.name for l in model.layers]
    k = names.index("gap") if "gap" in names else names.index("flatten")
    def chain(layers, shape):
        inp = keras.Input(shape)
        h = inp
        for layer in layers:
            h = layer(h)
        return keras.Model(inp, h)

    feat = chain(model.layers[:k + 1], model.input_shape[1:])
    head = chain(model.layers[k + 1:], tuple(feat.output.shape[1:]))
    return feat, head


def predict_mc_dropout(model, x, n_samples=config.MC_DROPOUT_SAMPLES, seed=config.SEED):
    """MC dropout on the classifier head: features are computed once, then the head is run
    `n_samples` times with dropout ACTIVE. Returns (mean probs, samples [T, N, out])."""
    keras.utils.set_random_seed(seed)
    feat, head = split_head(model)
    f = feat.predict(_batches(x), verbose=0)
    samples = np.stack([np.asarray(head(f, training=True)) for _ in range(n_samples)])
    return samples.mean(axis=0), samples


def run(task: str, name: str, data: dict) -> dict:
    model = keras.models.load_model(config.models_dir(task) / f"{name}.keras")
    out = {}
    for split in ["val", "test"]:
        x, y = data[split]
        out[f"y_{split}"] = y
        out[f"{split}_plain"] = as_matrix(predict_plain(model, x), task)
        out[f"{split}_tta"] = as_matrix(predict_tta(model, x), task)
        mean, samples = predict_mc_dropout(model, x)
        out[f"{split}_mc"] = as_matrix(mean, task)
        # Uncertainty = std of the predicted-class probability across MC samples.
        s = np.stack([as_matrix(t, task) for t in samples])          # (T, N, C)
        pred = out[f"{split}_mc"].argmax(1)
        out[f"{split}_mc_std"] = s[:, np.arange(len(pred)), pred].std(axis=0)
    config.preds_dir(task).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(config.preds_dir(task) / f"{name}.npz", **out)
    keras.backend.clear_session()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(config.CLASS_NAMES), default="binary")
    p.add_argument("--model", choices=config.MODEL_NAMES, required=True)
    p.add_argument("--data_dir", default=str(config.DEFAULT_DATA_DIR))
    a = p.parse_args()
    meta_size = keras.models.load_model(config.models_dir(a.task) / f"{a.model}.keras").input_shape[1]
    run(a.task, a.model, load_all(a.task, a.data_dir, meta_size))


if __name__ == "__main__":
    main()
