"""Streamlit demo: upload a chest X-ray -> calibrated prediction, uncertainty-based
referral flag, and a Grad-CAM heatmap.

    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import streamlit as st
from PIL import Image

from src import config
from src.gradcam import gradcam_heatmap, overlay
from src.inference import load_bundle, predict, to_input

st.set_page_config(page_title="Pneumonia Detection", page_icon="🫁", layout="wide")
st.title("🫁 Pneumonia Detection from Chest X-rays")
st.caption("Calibrated deep-learning classifier with uncertainty-based referral and Grad-CAM.")
st.warning("Educational project - not a medical device, not for clinical use.", icon="⚠️")

tasks = [t for t in config.CLASS_NAMES if (config.task_dir(t) / "results.json").exists()]
if not tasks:
    st.error(f"No results found in `{config.ARTIFACTS_DIR}`. Run `python -m src.pipeline` "
             "(or unzip the Kaggle artifacts) first.")
    st.stop()


@st.cache_resource(show_spinner="Loading model...")
def get_bundle(task):
    return load_bundle(task)


with st.sidebar:
    task = st.radio("Task", tasks, format_func={
        "binary": "Normal vs Pneumonia", "multiclass": "Normal / Bacterial / Viral"}.get)
    b = get_bundle(task)
    st.markdown(f"**Model:** {config.PRETTY[b.name]}")
    alpha = st.slider("Heatmap opacity", 0.0, 1.0, 0.4, 0.05)

up = st.file_uploader("Upload a chest X-ray (JPEG/PNG)", type=["jpg", "jpeg", "png"])
if up is None:
    st.info("Try an image from the test/ folder of the Kaggle dataset.")
    st.stop()

x = to_input(Image.open(up), b.img_size)
with st.spinner("Analysing..."):
    r = predict(b, x)
    heat, _ = gradcam_heatmap(b.model, x, class_idx=r["pred"] if task != "binary" else None)

c1, c2, c3 = st.columns([1, 1, 0.9])
c1.image(x[0].astype(np.uint8), caption="Input", width="stretch")
c2.image(overlay(x, heat, alpha), caption="Grad-CAM: regions driving the prediction",
         width="stretch")
with c3:
    st.subheader("Result")
    (st.success if r["label"] == "NORMAL" else st.error)(f"**{r['label']}**")
    for cls, p in r["probs"].items():
        st.write(f"{cls}: {p:.1%}")
        st.progress(p)
    st.metric("Uncertainty (MC-dropout std)", f"{r['uncertainty']:.3f}")
    if r["refer"]:
        st.warning("High uncertainty - this case would be referred to a radiologist.", icon="🩺")
    else:
        st.caption("Confidence is within the range the model handles autonomously.")
