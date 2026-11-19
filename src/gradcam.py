"""Grad-CAM (Selvaraju et al., ICCV 2017) for any model in the zoo.

Models are flat Sequential stacks ([preprocess] -> backbone | conv blocks -> head), so we
run the layers one by one inside a GradientTape and grab the target feature map.

    python -m src.gradcam --task binary --model densenet121
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src import config
from src.models import gradcam_layer_name


def gradcam_heatmap(model, x: np.ndarray, class_idx: int | None = None):
    """x: (1, S, S, 3) float in [0,255]. Returns (heatmap [S,S] in [0,1], probs [C])."""
    target = gradcam_layer_name(model)
    h = tf.convert_to_tensor(x, tf.float32)
    fmap = None
    with tf.GradientTape() as tape:
        for layer in model.layers:
            h = layer(h, training=False)
            if layer.name == target:
                fmap = h
                tape.watch(fmap)       # works even when the layer is frozen
        probs = h[0]
        if probs.shape[-1] == 1:       # binary sigmoid: explain the PNEUMONIA score
            score = probs[0]
        else:
            idx = int(tf.argmax(probs)) if class_idx is None else class_idx
            score = probs[idx]
    grads = tape.gradient(score, fmap)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.reduce_sum(fmap[0] * weights, axis=-1))
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], x.shape[1:3], method="bilinear")[..., 0]
    p = probs.numpy()
    p = np.array([1 - p[0], p[0]]) if p.shape[-1] == 1 else p
    return cam.numpy(), p


def overlay(x: np.ndarray, heat: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    img = x[0].astype("float32") / 255.0
    colored = matplotlib.colormaps["jet"](heat)[..., :3]
    return (np.clip((1 - alpha) * img + alpha * colored, 0, 1) * 255).astype("uint8")


def gallery(task: str, name: str, x_test, y_test, n_per_class=3, seed=config.SEED, out=None,
            threshold: float | None = None):
    """Grid of test images: correct and incorrect examples per class, with Grad-CAM."""
    import keras
    model = keras.models.load_model(config.models_dir(task) / f"{name}.keras")
    classes = config.CLASS_NAMES[task]
    rng = np.random.default_rng(seed)
    picks = []
    for c in range(len(classes)):
        idx = np.where(y_test == c)[0]
        picks += list(rng.choice(idx, min(n_per_class, len(idx)), replace=False))
    cols = len(picks)
    fig, axes = plt.subplots(2, cols, figsize=(2.3 * cols, 5.2))
    for j, i in enumerate(picks):
        x = np.repeat(x_test[i][None, ..., None].astype("float32"), 3, -1)
        heat, p = gradcam_heatmap(model, x)
        pred = int(p[1] >= threshold) if (threshold is not None and len(p) == 2) else int(p.argmax())
        axes[0, j].imshow(x_test[i], cmap="gray")
        axes[0, j].set_title(f"true: {classes[y_test[i]]}", fontsize=8)
        axes[1, j].imshow(overlay(x, heat))
        axes[1, j].set_title(f"pred: {classes[pred]} {p[pred]:.2f}", fontsize=8,
                             color="#008300" if pred == y_test[i] else "#e34948")
        for ax in axes[:, j]:
            ax.axis("off")
    fig.suptitle(f"Grad-CAM - {config.PRETTY[name]} ({task})", fontsize=11)
    fig.tight_layout()
    out = out or config.figures_dir(task) / f"gradcam_{name}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


if __name__ == "__main__":
    from src.data import load_all
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(config.CLASS_NAMES), default="binary")
    p.add_argument("--model", choices=config.MODEL_NAMES, required=True)
    p.add_argument("--data_dir", default=str(config.DEFAULT_DATA_DIR))
    a = p.parse_args()
    d = load_all(a.task, a.data_dir)
    print(gallery(a.task, a.model, *d["test"]))
