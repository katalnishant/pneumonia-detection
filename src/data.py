"""Data: discovery, patient-level splitting, cached image arrays, tf.data pipelines.

* The official val/ folder has only 16 images, so train/ + val/ are merged and re-split
  BY PATIENT (StratifiedGroupKFold): no person appears in both train and validation.
  Stratification uses the 3-class label, so the binary and 3-class tasks share the
  exact same split. The official test/ folder stays untouched.
* Every image is decoded and resized ONCE and cached as uint8 (~250 MB for 224 px),
  so all models train from memory instead of re-decoding JPEGs each epoch.
"""
from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from keras import layers
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

from src import config

IMG_EXTS = {".jpeg", ".jpg", ".png"}
AUTOTUNE = tf.data.AUTOTUNE


# ---------------------------------------------------------------------------
# Discovery and labels
# ---------------------------------------------------------------------------
def subtype_label(path: str) -> int | None:
    """0 = NORMAL, 1 = BACTERIAL, 2 = VIRAL (parsed from folder + file name)."""
    p = Path(path)
    if p.parent.name.upper() == "NORMAL":
        return 0
    name = p.name.lower()
    if "bacteria" in name:
        return 1
    if "virus" in name:
        return 2
    return None


def to_task_labels(sub_labels, task: str) -> np.ndarray:
    sub = np.asarray(sub_labels)
    return (sub > 0).astype("int64") if task == "binary" else sub.astype("int64")


def list_images(split_dir: Path) -> tuple[list[str], list[int]]:
    paths, subs = [], []
    for cls in ["NORMAL", "PNEUMONIA"]:
        cls_dir = split_dir / cls
        if not cls_dir.exists():
            continue
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                s = subtype_label(str(p))
                if s is not None:
                    paths.append(str(p))
                    subs.append(s)
    return paths, subs


def patient_id(path: str) -> str:
    """person1000_virus_1681.jpeg -> person1000 ; IM-0115-0001.jpeg -> IM-0115 ;
    NORMAL2-IM-0329-0001.jpeg -> NORMAL2-IM-0329 ; otherwise the file name."""
    name = Path(path).name
    m = re.match(r"(person\d+)_", name) or re.match(r"(.*IM-\d+)-", name)
    return f"{Path(path).parent.name}/{m.group(1) if m else Path(path).stem}"


def resolve_data_dir(data_dir) -> Path:
    data_dir = Path(data_dir)
    for cand in (data_dir, data_dir / "chest_xray"):
        if (cand / "train").exists() and (cand / "test").exists():
            return cand
    raise FileNotFoundError(f"Could not find train/ and test/ under {data_dir}")


def make_splits(data_dir=config.DEFAULT_DATA_DIR, val_fraction=config.VAL_FRACTION,
                seed=config.SEED) -> dict:
    """Returns {"train"|"val"|"test": {"paths": [...], "sub": [...] (3-class labels)}}."""
    root = resolve_data_dir(data_dir)
    tr_p, tr_s = list_images(root / "train")
    va_p, va_s = list_images(root / "val")
    pool_p, pool_s = np.array(tr_p + va_p), np.array(tr_s + va_s)
    groups = np.array([patient_id(p) for p in pool_p])

    sgkf = StratifiedGroupKFold(n_splits=max(2, round(1 / val_fraction)), shuffle=True,
                                random_state=seed)
    tr_idx, va_idx = next(sgkf.split(pool_p, pool_s, groups))
    assert not set(groups[tr_idx]) & set(groups[va_idx]), "patient leakage!"

    te_p, te_s = list_images(root / "test")
    return {
        "train": {"paths": pool_p[tr_idx].tolist(), "sub": pool_s[tr_idx].tolist()},
        "val": {"paths": pool_p[va_idx].tolist(), "sub": pool_s[va_idx].tolist()},
        "test": {"paths": te_p, "sub": te_s},
    }


def describe_splits(splits: dict) -> str:
    lines = [f"{'split':>6} {'total':>6} {'NORMAL':>7} {'BACTERIAL':>10} {'VIRAL':>6}"]
    for name, d in splits.items():
        c = np.bincount(d["sub"], minlength=3)
        lines.append(f"{name:>6} {len(d['sub']):>6} {c[0]:>7} {c[1]:>10} {c[2]:>6}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Image loading (shared by training and inference => identical preprocessing)
# ---------------------------------------------------------------------------
def load_gray(src, img_size: int = config.IMG_SIZE) -> np.ndarray:
    """Path or PIL image -> uint8 array (img_size, img_size), grayscale."""
    img = src if isinstance(src, Image.Image) else Image.open(src)
    return np.asarray(img.convert("L").resize((img_size, img_size), Image.BILINEAR),
                      dtype=np.uint8)


def load_split_arrays(paths: list[str], img_size: int, name: str) -> np.ndarray:
    """Decode + resize all images once; cache to disk as .npy (keyed by the path list)."""
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(json.dumps([img_size] + list(paths)).encode()).hexdigest()[:10]
    cache = config.CACHE_DIR / f"{name}_{img_size}_{key}.npy"
    if cache.exists():
        return np.load(cache)
    with ThreadPoolExecutor(max_workers=8) as ex:
        arr = np.stack(list(ex.map(lambda p: load_gray(p, img_size), paths)))
    np.save(cache, arr)
    return arr


def load_all(task: str, data_dir=config.DEFAULT_DATA_DIR, img_size=config.IMG_SIZE,
             limit: int | None = None) -> dict:
    """Return {"train"|"val"|"test": (x_uint8 [N,S,S], y [N])} for `task`.
    `limit` keeps a small stratified subset per split (for quick smoke runs)."""
    splits = make_splits(data_dir)
    out = {"paths": {}}
    for name, d in splits.items():
        paths, sub = d["paths"], np.asarray(d["sub"])
        if limit:
            rng = np.random.default_rng(config.SEED)
            idx = np.sort(rng.permutation(len(paths))[:limit])
            paths, sub = [paths[i] for i in idx], sub[idx]
        x = load_split_arrays(paths, img_size, name)
        out[name] = (x, to_task_labels(sub, task))
        out["paths"][name] = paths
    return out


def get_class_weights(y) -> dict[int, float]:
    classes = np.unique(y)
    w = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(v) for c, v in zip(classes, w)}


# ---------------------------------------------------------------------------
# tf.data
# ---------------------------------------------------------------------------
def build_augmenter() -> keras.Sequential:
    """Mild, anatomically plausible augmentation. No horizontal flip: the heart is on the
    left, so a mirrored chest X-ray is not a realistic image."""
    return keras.Sequential([
        layers.RandomRotation(0.03),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomContrast(0.15),
    ], name="augmentation")


def to_model_input(x_uint8):
    """uint8 (B,S,S) -> float32 (B,S,S,3) in [0,255]: the input format of every model."""
    x = tf.cast(x_uint8, tf.float32)[..., None]
    return tf.repeat(x, 3, axis=-1)


def encode_labels(y, task: str, n_classes: int):
    if task == "binary":
        return tf.cast(y, tf.float32)
    return tf.one_hot(y, n_classes)


def make_dataset(x, y, task: str, batch_size=config.BATCH_SIZE, training=False,
                 seed=config.SEED) -> tf.data.Dataset:
    n_classes = len(config.CLASS_NAMES[task])
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(len(x), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    aug = build_augmenter() if training else None

    def prep(xb, yb):
        xb = to_model_input(xb)
        if aug is not None:
            xb = tf.clip_by_value(aug(xb, training=True), 0.0, 255.0)
        return xb, encode_labels(yb, task, n_classes)

    return ds.map(prep, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
