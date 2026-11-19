"""Generate notebooks/pneumonia_kaggle.ipynb: a self-contained Kaggle notebook that writes
the src/ package to disk (one readable cell per module), runs the whole study on a free
GPU and displays the results. Regenerate it after changing anything in src/:

    python scripts/build_kaggle_notebook.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
REPO = "https://github.com/katalnishant/pneumonia-detection"
SRC = ["__init__", "config", "data", "models", "train", "predict", "analyze", "gradcam",
       "inference", "pipeline"]
KAGGLE_PROJECT = "/kaggle/working/pneumonia-detection"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("# 🫁 Pneumonia Detection from Chest X-rays: a 6-model benchmark with trustworthy-AI analysis\n\n"
       f"Code: {REPO}\n\n"
       "**What this notebook does** (`python -m src.pipeline`):\n"
       "1. **Patient-level** re-split of the dataset (no patient in both train and validation; "
       "the official 16-image val folder is too small)\n"
       "2. Two tasks: **Normal vs Pneumonia** and **Normal / Bacterial / Viral**\n"
       "3. Six models: custom CNN, the original *VGG16 v1* recipe (frozen base + Flatten head), "
       "and two-phase fine-tuned **VGG16, ResNet50, DenseNet121, EfficientNetB0**\n"
       "4. **Test-time augmentation** and a **top-3 ensemble**\n"
       "5. **Calibration** (temperature scaling, ECE), **MC-dropout uncertainty**, "
       "**radiologist referral** of the most uncertain cases, **95% bootstrap CIs**\n"
       "6. **Grad-CAM** explanations\n\n"
       "Everything that's tuned (thresholds, temperatures, referral cut-offs, ensemble members) is "
       "chosen on the **validation** set. The test set is only used for the final numbers.\n\n"
       "**Setup:** dataset `paultimothymooney/chest-xray-pneumonia`, Accelerator **GPU**, "
       "Internet **on** (to download ImageNet weights)."),
    md("### ⚠️ Always use **Run All** (or run the cells top to bottom)\n"
       "The next cell creates the project folder. If the session restarts, `/kaggle/working` is "
       "wiped, so start again from here."),
    code("import os\nPROJECT = '/kaggle/working/pneumonia-detection'\n"
         "os.makedirs(PROJECT + '/src', exist_ok=True)\nos.chdir(PROJECT)\n"
         "!nvidia-smi -L\nimport tensorflow as tf, keras\n"
         "print('TF', tf.__version__, '| Keras', keras.__version__, '| GPU:', "
         "tf.config.list_physical_devices('GPU'))"),
    md("## 1. Source code\nEach cell writes one module of the `src/` package."),
]
for name in SRC:
    body = (ROOT / "src" / f"{name}.py").read_text()
    cells.append(code(f"%%writefile {KAGGLE_PROJECT}/src/{name}.py\n{body}"))

cells += [
    md("## 2. Run the study\nSet `QUICK = True` for a ~10-minute test on a small subset. "
       "Set it back to `False` for the real run (about 3–4 h on a T4). Already-trained models are "
       "skipped, so re-running resumes after an interruption."),
    code("QUICK = False   # True = quick test, False = full study\n"
         f"%cd {KAGGLE_PROJECT}\n"
         "FLAG = '--quick' if QUICK else ''\n"
         "!python -m src.pipeline $FLAG"),
    md("## 3. Results"),
    code(f"%cd {KAGGLE_PROJECT}\n"
         "from IPython.display import Image, Markdown, display\n"
         "display(Image('artifacts/dataset_overview.png'))\n"
         "for task in ['binary', 'multiclass']:\n"
         "    display(Markdown(f'### Task: {task}'))\n"
         "    display(Markdown(open(f'artifacts/{task}/results.md').read()))\n"
         "    for f in ['model_comparison', 'roc_curves', 'reliability', 'referral_curve']:\n"
         "        display(Image(f'artifacts/{task}/figures/{f}.png'))"),
    md("## 4. Explainability (Grad-CAM) and confusion matrices"),
    code("import glob\nfor task in ['binary', 'multiclass']:\n"
         "    for f in sorted(glob.glob(f'artifacts/{task}/figures/gradcam_*.png')) + "
         "sorted(glob.glob(f'artifacts/{task}/figures/confusion_*.png')):\n"
         "        print(f); display(Image(f))"),
    md("## 5. Package the artifacts\n`artifacts.zip` (in Output) holds models, metrics and figures. "
       "Unzip it in the repo root to run the Streamlit app."),
    code("!rm -rf artifacts/cache\n!zip -qr /kaggle/working/artifacts.zip artifacts "
         "&& ls -lh /kaggle/working/artifacts.zip"),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"}})
out = ROOT / "notebooks" / "pneumonia_kaggle.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print(f"Wrote {out}")
