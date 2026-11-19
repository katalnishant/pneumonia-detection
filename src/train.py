"""Train one model on one task.

    python -m src.train --task binary --model densenet121
    python -m src.train --task multiclass --model efficientnetb0 --quick   # tiny smoke run
"""
from __future__ import annotations

import argparse
import json
import time

import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src.data import get_class_weights, load_all, make_dataset
from src.models import build_model, compile_model, set_phase


def callbacks(path, patience, best_so_far=None):
    return [
        keras.callbacks.ModelCheckpoint(path, monitor="val_auc", mode="max", save_best_only=True,
                                        initial_value_threshold=best_so_far, verbose=0),
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2,
                                          min_lr=1e-7, verbose=1),
    ]


def merge(*hists) -> dict:
    out: dict = {}
    for h in hists:
        for k, v in h.history.items():
            out.setdefault(k, []).extend(float(x) for x in v)
    return out


def plot_history(hist, name, task, boundary=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, m in zip(axes, ["loss", "accuracy", "auc"]):
        ax.plot(hist[m], label="train")
        ax.plot(hist[f"val_{m}"], label="val")
        if boundary:
            ax.axvline(boundary - 0.5, color="gray", ls="--", label="fine-tuning starts")
        ax.set(title=m, xlabel="epoch")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle(f"{config.PRETTY[name]} - {task}")
    fig.tight_layout()
    config.figures_dir(task).mkdir(parents=True, exist_ok=True)
    fig.savefig(config.figures_dir(task) / f"history_{name}.png", dpi=110)
    plt.close(fig)


def train_model(name: str, task: str, data: dict, img_size=config.IMG_SIZE,
                batch_size=config.BATCH_SIZE, weights="imagenet", epoch_scale: float = 1.0,
                seed=config.SEED) -> dict:
    """Train `name` on `task` using pre-loaded `data` (from load_all). Returns metadata."""
    keras.utils.set_random_seed(seed)
    mdir = config.models_dir(task)
    mdir.mkdir(parents=True, exist_ok=True)
    path = mdir / f"{name}.keras"
    ep = lambda n: max(1, round(n * epoch_scale))

    x_tr, y_tr = data["train"]
    train_ds = make_dataset(x_tr, y_tr, task, batch_size, training=True, seed=seed)
    val_ds = make_dataset(*data["val"], task, batch_size)
    cw = get_class_weights(y_tr)
    model = build_model(name, task, img_size, weights)
    t0, boundary = time.time(), None

    if name == "cnn":
        compile_model(model, task, config.CNN_LR)
        h = model.fit(train_ds, validation_data=val_ds, epochs=ep(config.CNN_EPOCHS),
                      class_weight=cw, callbacks=callbacks(path, config.CNN_PATIENCE), verbose=2)
        hist = merge(h)
    elif name == "vgg16_v1":
        # Original recipe: frozen base, Adam(1e-4), no class weights, no fine-tuning.
        set_phase(model, name, "head")
        compile_model(model, task, config.V1_LR)
        h = model.fit(train_ds, validation_data=val_ds, epochs=ep(config.V1_EPOCHS),
                      callbacks=callbacks(path, config.PATIENCE), verbose=2)
        hist = merge(h)
    else:
        set_phase(model, name, "head")
        compile_model(model, task, config.HEAD_LR)
        h1 = model.fit(train_ds, validation_data=val_ds, epochs=ep(config.HEAD_EPOCHS),
                       class_weight=cw, callbacks=callbacks(path, config.PATIENCE), verbose=2)
        boundary = len(h1.history["loss"])
        set_phase(model, name, "finetune")
        compile_model(model, task, config.FINETUNE_LR)   # recompile after changing trainable
        h2 = model.fit(train_ds, validation_data=val_ds, initial_epoch=boundary,
                       epochs=boundary + ep(config.FINETUNE_EPOCHS), class_weight=cw,
                       callbacks=callbacks(path, config.PATIENCE, max(h1.history["val_auc"])),
                       verbose=2)
        hist = merge(h1, h2)

    meta = {"model": name, "task": task, "img_size": img_size,
            "class_names": config.CLASS_NAMES[task], "best_val_auc": max(hist["val_auc"]),
            "epochs_run": len(hist["loss"]), "train_minutes": round((time.time() - t0) / 60, 1),
            "params": int(model.count_params())}
    (mdir / f"{name}.json").write_text(json.dumps(meta, indent=2))
    (mdir / f"history_{name}.json").write_text(json.dumps(hist))
    plot_history(hist, name, task, boundary)
    print(f"[{task}/{name}] best val AUC {meta['best_val_auc']:.4f} in {meta['train_minutes']} min")
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(config.CLASS_NAMES), default="binary")
    p.add_argument("--model", choices=config.MODEL_NAMES, required=True)
    p.add_argument("--data_dir", default=str(config.DEFAULT_DATA_DIR))
    p.add_argument("--img_size", type=int, default=config.IMG_SIZE)
    p.add_argument("--weights", default="imagenet", help="'imagenet' or 'none'")
    p.add_argument("--quick", action="store_true", help="tiny subset + 1-2 epochs (smoke test)")
    a = p.parse_args()
    data = load_all(a.task, a.data_dir, a.img_size, limit=200 if a.quick else None)
    train_model(a.model, a.task, data, a.img_size,
                weights=None if a.weights == "none" else a.weights,
                epoch_scale=0.1 if a.quick else 1.0)


if __name__ == "__main__":
    main()
