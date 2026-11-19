"""Central configuration. Every script reads its defaults from here."""
import os
from pathlib import Path

# ---- Paths -------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
# Kaggle path of "Chest X-Ray Images (Pneumonia)" (paultimothymooney/chest-xray-pneumonia).
DEFAULT_DATA_DIR = Path(os.environ.get("PNEUMO_DATA", "/kaggle/input/chest-xray-pneumonia/chest_xray"))
ARTIFACTS_DIR = Path(os.environ.get("PNEUMO_ARTIFACTS", ROOT_DIR / "artifacts"))
CACHE_DIR = ARTIFACTS_DIR / "cache"


def task_dir(task: str) -> Path:
    return ARTIFACTS_DIR / task


def models_dir(task: str) -> Path:
    return task_dir(task) / "models"


def preds_dir(task: str) -> Path:
    return task_dir(task) / "preds"


def figures_dir(task: str) -> Path:
    return task_dir(task) / "figures"


# ---- Tasks -------------------------------------------------------------------
# binary:     NORMAL vs PNEUMONIA
# multiclass: NORMAL vs BACTERIAL vs VIRAL pneumonia (subtype parsed from the file names)
CLASS_NAMES = {
    "binary": ["NORMAL", "PNEUMONIA"],
    "multiclass": ["NORMAL", "BACTERIAL", "VIRAL"],
}

# ---- Models ------------------------------------------------------------------
MODEL_NAMES = ["cnn", "vgg16_v1", "vgg16", "resnet50", "densenet121", "efficientnetb0"]
PRETTY = {
    "cnn": "Custom CNN (from scratch)",
    "vgg16_v1": "VGG16 v1 (frozen + Flatten head)",
    "vgg16": "VGG16 (fine-tuned)",
    "resnet50": "ResNet50 (fine-tuned)",
    "densenet121": "DenseNet121 (fine-tuned)",
    "efficientnetb0": "EfficientNetB0 (fine-tuned)",
    "ensemble": "Ensemble (top-3, TTA)",
}

# ---- Data --------------------------------------------------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
VAL_FRACTION = 0.15
SEED = 42

# ---- Training ----------------------------------------------------------------
CNN_EPOCHS = 30
CNN_LR = 1e-3
CNN_PATIENCE = 6

HEAD_EPOCHS = 8         # phase 1: frozen backbone, train the new head
FINETUNE_EPOCHS = 12    # phase 2: unfreeze the last block(s), small LR
HEAD_LR = 1e-3
FINETUNE_LR = 1e-5
PATIENCE = 4

V1_EPOCHS = 15          # "vgg16_v1" reproduces the original notebook: frozen base, Adam 1e-4
V1_LR = 1e-4

# ---- Analysis ----------------------------------------------------------------
ENSEMBLE_TOP_K = 3      # ensemble = top-k models ranked by VALIDATION AUC
MC_DROPOUT_SAMPLES = 30
N_BOOTSTRAP = 1000
REFERRAL_RATE = 0.10    # share of most-uncertain cases sent to a radiologist in the headline number
