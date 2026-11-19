"""Run the whole study end to end: EDA -> train every model -> predict (plain, TTA,
MC dropout) -> analysis (calibration, ensemble, CIs, referral) -> Grad-CAM galleries.

    python -m src.pipeline                                   # both tasks, all models
    python -m src.pipeline --tasks binary --models cnn densenet121
    python -m src.pipeline --quick                           # 5-minute smoke run
Already-trained models are skipped unless --force is given, so a crashed run resumes.
"""
from __future__ import annotations

import argparse
import gc
import json
import time

import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import analyze, config, gradcam, predict
from src.data import describe_splits, load_all, make_splits
from src.train import train_model


def eda(data_dir, out_dir):
    """Dataset overview figure: class counts per split + example images per class."""
    splits = make_splits(data_dir)
    print(describe_splits(splits))
    out_dir.mkdir(parents=True, exist_ok=True)
    names = config.CLASS_NAMES["multiclass"]
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig = plt.figure(figsize=(14, 6.5))
    ax = fig.add_subplot(1, 4, 1)
    x = np.arange(3)
    for i, c in enumerate(names):
        counts = [np.sum(np.asarray(splits[s]["sub"]) == i) for s in ["train", "val", "test"]]
        ax.bar(x + (i - 1) * 0.27, counts, 0.25, color=colors[i], label=c)
    ax.set_xticks(x, ["train", "val", "test"])
    ax.set_title("Images per class", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    tr = splits["train"]
    for i, c in enumerate(names):
        idx = [j for j, s in enumerate(tr["sub"]) if s == i][:2]
        for k, j in enumerate(idx):
            a = fig.add_subplot(2, 4, 2 + i + 4 * k)
            a.imshow(plt.imread(tr["paths"][j]), cmap="gray")
            a.set_title(c, fontsize=9)
            a.axis("off")
    fig.suptitle("Chest X-Ray Images (Pneumonia) - patient-level split", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "dataset_overview.png", dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["binary", "multiclass"],
                   choices=list(config.CLASS_NAMES))
    p.add_argument("--models", nargs="+", default=config.MODEL_NAMES, choices=config.MODEL_NAMES)
    p.add_argument("--data_dir", default=str(config.DEFAULT_DATA_DIR))
    p.add_argument("--img_size", type=int, default=config.IMG_SIZE)
    p.add_argument("--weights", default="imagenet")
    p.add_argument("--quick", action="store_true", help="small subset, ~10%% epochs")
    p.add_argument("--force", action="store_true", help="retrain models that already exist")
    a = p.parse_args()
    weights = None if a.weights == "none" else a.weights
    t_start = time.time()

    eda(a.data_dir, config.ARTIFACTS_DIR)
    for task in a.tasks:
        print(f"\n################ TASK: {task} ################")
        data = load_all(task, a.data_dir, a.img_size, limit=300 if a.quick else None)
        for name in a.models:
            ckpt = config.models_dir(task) / f"{name}.keras"
            if ckpt.exists() and not a.force:
                print(f"[{task}/{name}] already trained - skipping")
            else:
                print(f"\n======== [{task}] training {config.PRETTY[name]} ========")
                train_model(name, task, data, a.img_size, weights=weights,
                            epoch_scale=0.1 if a.quick else 1.0)
            keras.backend.clear_session()
            gc.collect()
            predict.run(task, name, data)
            gc.collect()

        analyze.main(task)
        res = json.loads((config.task_dir(task) / "results.json").read_text())
        best = res["best_single_model"]
        thr = {m["name"]: m["threshold"] for m in res["models"]}
        for name in dict.fromkeys([best, "cnn"]):
            if name in a.models:
                gradcam.gallery(task, name, *data["test"], threshold=thr[name])
        del data
        gc.collect()
    print(f"\nAll done in {(time.time() - t_start) / 60:.1f} min. Artifacts: {config.ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
