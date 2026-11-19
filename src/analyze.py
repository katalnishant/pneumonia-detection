"""Analysis of all trained models for one task (run after src.predict for every model).

For every model and for a top-k ensemble:
  * test-time augmentation (TTA) vs plain predictions
  * temperature scaling fitted on VALIDATION -> calibrated probabilities, ECE before/after
  * binary: decision threshold tuned on validation (Youden's J)
  * test metrics with 95% bootstrap confidence intervals
  * uncertainty-based referral: the most uncertain cases (threshold chosen on validation)
    are routed to a radiologist; we report accuracy on the cases the model keeps
Outputs: artifacts/<task>/results.json, results.md and figures/*.png

    python -m src.analyze --task binary
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve, ConfusionMatrixDisplay)

from src import config

# Reference categorical palette (fixed order, validated for colour-vision deficiency).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
NEUTRAL, INK, MUTED = "#b8b7b0", "#0b0b0b", "#52514e"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.titleweight": "bold", "font.size": 10})


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def auc(y, P):
    if P.shape[1] == 2:
        return roc_auc_score(y, P[:, 1])
    return roc_auc_score(y, P, multi_class="ovr", average="macro")


def ece(y, P, n_bins=15) -> float:
    """Expected calibration error of the top-class confidence."""
    conf, pred = P.max(1), P.argmax(1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def fit_temperature(y, P) -> float:
    logp = np.log(np.clip(P, 1e-7, 1))

    def nll(t):
        z = logp / t
        z -= z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -lp[np.arange(len(y)), y].mean()
    return float(minimize_scalar(nll, bounds=(0.05, 10), method="bounded").x)


def apply_temperature(P, t):
    z = np.log(np.clip(P, 1e-7, 1)) / t
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def youden(y, p) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    return float(np.clip(thr[np.argmax(tpr - fpr)], 0, 1))


def predict_labels(P, thr):
    return (P[:, 1] >= thr).astype(int) if P.shape[1] == 2 else P.argmax(1)


def point_metrics(y, P, thr) -> dict:
    yp = predict_labels(P, thr)
    if P.shape[1] == 2:
        tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
        return {"accuracy": accuracy_score(y, yp), "auc": auc(y, P),
                "f1": f1_score(y, yp), "precision": precision_score(y, yp, zero_division=0),
                "sensitivity": recall_score(y, yp), "specificity": tn / (tn + fp)}
    rec = recall_score(y, yp, average=None, labels=list(range(P.shape[1])), zero_division=0)
    out = {"accuracy": accuracy_score(y, yp), "auc": auc(y, P),
           "f1": f1_score(y, yp, average="macro")}
    out.update({f"recall_{c.lower()}": float(r)
                for c, r in zip(config.CLASS_NAMES["multiclass"], rec)})
    return out


def bootstrap_ci(y, P, thr, n=config.N_BOOTSTRAP, seed=config.SEED) -> dict:
    rng = np.random.default_rng(seed)
    keys = ["accuracy", "auc", "f1"]
    samples = {k: [] for k in keys}
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < P.shape[1]:
            continue
        m = point_metrics(y[idx], P[idx], thr)
        for k in keys:
            samples[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in samples.items()}


def referral(y_val, P_val, u_val, y_te, P_te, u_te, thr, rate=config.REFERRAL_RATE) -> dict:
    """Refer cases with uncertainty above the validation (1-rate) quantile."""
    cut = np.quantile(u_val, 1 - rate)
    keep = u_te <= cut
    yp = predict_labels(P_te, thr)
    correct = yp == y_te
    return {"uncertainty_cutoff": float(cut), "referred_fraction": float(1 - keep.mean()),
            "accuracy_all": float(correct.mean()), "accuracy_kept": float(correct[keep].mean()),
            "errors_caught": int((~correct & ~keep).sum()), "errors_total": int((~correct).sum())}


def coverage_curve(y, P, u, thr, coverages=np.linspace(0.5, 1.0, 26)):
    order = np.argsort(u)                        # most certain first
    correct = (predict_labels(P, thr) == y)[order]
    return coverages, [correct[: max(1, int(round(c * len(y))))].mean() for c in coverages]


# ---------------------------------------------------------------------------
def evaluate_model(name, d, task) -> dict:
    """d holds y_val, y_test and val/test probability matrices: *_plain, *_tta, *_unc."""
    y_val, y_te = d["y_val"], d["y_test"]
    t = fit_temperature(y_val, d["val_tta"])
    Pv, Pt = apply_temperature(d["val_tta"], t), apply_temperature(d["test_tta"], t)
    thr = youden(y_val, Pv[:, 1]) if task == "binary" else None
    res = {
        "name": name, "pretty": config.PRETTY[name],
        "val_auc": auc(y_val, d["val_tta"]),
        "test_auc_plain": auc(y_te, d["test_plain"]), "test_auc_tta": auc(y_te, d["test_tta"]),
        "temperature": t, "threshold": thr,
        "ece_before": ece(y_te, d["test_tta"]), "ece_after": ece(y_te, Pt),
        "test": point_metrics(y_te, Pt, thr),
        "test_at_0.5": point_metrics(y_te, d["test_plain"], 0.5) if task == "binary" else None,
        "ci95": bootstrap_ci(y_te, Pt, thr),
        "referral": referral(y_val, Pv, d["val_unc"], y_te, Pt, d["test_unc"], thr),
    }
    res["_P_test"], res["_P_val"] = Pt, Pv
    return res


def build_ensemble(preds: dict, ranked: list[str], k: int) -> dict:
    members = ranked[:k]
    d = {"y_val": preds[members[0]]["y_val"], "y_test": preds[members[0]]["y_test"]}
    for s in ["val", "test"]:
        stack = np.stack([preds[m][f"{s}_tta"] for m in members])         # (K, N, C)
        d[f"{s}_tta"] = stack.mean(0)
        d[f"{s}_plain"] = np.stack([preds[m][f"{s}_plain"] for m in members]).mean(0)
        pred = d[f"{s}_tta"].argmax(1)
        # Uncertainty = disagreement between members on the predicted class.
        d[f"{s}_unc"] = stack[:, np.arange(stack.shape[1]), pred].std(0)
    d["members"] = members
    return d


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_comparison(results, task, path):
    names = [r["name"] for r in results]
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.55 * len(names) + 1.8), sharey=True)
    best = max(results, key=lambda r: r["val_auc"])["name"]
    for ax, key, label in [(axes[0], "auc", "Test ROC-AUC"),
                           (axes[1], "f1", "Test F1" if task == "binary" else "Test macro-F1")]:
        vals = [r["test"][key] for r in results]
        lo = [r["test"][key] - r["ci95"][key][0] for r in results]
        hi = [r["ci95"][key][1] - r["test"][key] for r in results]
        colors = [SERIES[0] if n in ("ensemble", best) else NEUTRAL for n in names]
        y = np.arange(len(names))
        ax.barh(y, vals, color=colors, height=0.6, xerr=[lo, hi], ecolor=MUTED, capsize=3)
        for yi, v in zip(y, vals):
            ax.text(0.505, yi, f"{v:.3f}", va="center", ha="left", color="white", fontsize=9,
                    fontweight="bold")
        ax.set_xlim(0.5, 1.0)
        ax.set_title(label)
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(np.arange(len(names)), [config.PRETTY[n] for n in names])
    axes[0].invert_yaxis()
    fig.suptitle(f"Model comparison on the held-out test set ({task}) - bars show 95% bootstrap CI",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_roc(results, y_te, task, path):
    show = ["cnn", "vgg16_v1", max([r for r in results if r["name"] != "ensemble"],
                                    key=lambda r: r["val_auc"])["name"], "ensemble"]
    show = list(dict.fromkeys(n for n in show if any(r["name"] == n for r in results)))
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, n in enumerate(show):
        r = next(r for r in results if r["name"] == n)
        P = r["_P_test"]
        if task == "binary":
            fpr, tpr, _ = roc_curve(y_te, P[:, 1])
        else:  # micro-average one-vs-rest
            Y = np.eye(P.shape[1])[y_te]
            fpr, tpr, _ = roc_curve(Y.ravel(), P.ravel())
        ax.plot(fpr, tpr, color=SERIES[i], lw=2, label=f"{config.PRETTY[n]} (AUC {r['test']['auc']:.3f})")
    ax.plot([0, 1], [0, 1], color=NEUTRAL, ls="--", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title=f"ROC curves - test set ({task})")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_confusion(r, y_te, task, path):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    yp = predict_labels(r["_P_test"], r["threshold"])
    n = len(config.CLASS_NAMES[task])
    ConfusionMatrixDisplay(confusion_matrix(y_te, yp, labels=list(range(n))),
                           display_labels=config.CLASS_NAMES[task]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{config.PRETTY[r['name']]}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_reliability(r, d, path, n_bins=10):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    bins = np.linspace(0, 1, n_bins + 1)
    for i, (P, label) in enumerate([(d["test_tta"], f"before (ECE {r['ece_before']:.3f})"),
                                    (r["_P_test"], f"after temperature scaling (ECE {r['ece_after']:.3f})")]):
        conf, correct = P.max(1), P.argmax(1) == d["y_test"]
        xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 5:
                xs.append(conf[m].mean())
                ys.append(correct[m].mean())
        ax.plot(xs, ys, marker="o", ms=6, lw=2, color=SERIES[i], label=label)
    ax.plot([0, 1], [0, 1], color=NEUTRAL, ls="--", lw=1, label="perfect calibration")
    ax.set(xlabel="Predicted confidence", ylabel="Observed accuracy", xlim=(0.4, 1), ylim=(0.4, 1),
           title=f"Reliability - {config.PRETTY[r['name']]}")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_referral(curves, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, (label, cov, acc) in enumerate(curves):
        baseline = label.startswith("Random")
        ax.plot(cov * 100, np.array(acc) * 100, lw=1.5 if baseline else 2,
                ls="--" if baseline else "-", color=NEUTRAL if baseline else SERIES[i], label=label)
    ax.axvline(100 * (1 - config.REFERRAL_RATE), color=NEUTRAL, ls="--", lw=1)
    ax.text(100 * (1 - config.REFERRAL_RATE) - 0.5, ax.get_ylim()[0] + 0.3,
            f"refer {config.REFERRAL_RATE:.0%}", ha="right", color=MUTED, fontsize=8)
    ax.invert_xaxis()
    ax.set(xlabel="Coverage: % of test cases the model decides (rest referred to a radiologist)",
           ylabel="Accuracy on decided cases (%)", title="Uncertainty-based referral")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
def to_markdown(results, task, ens_members) -> str:
    f1 = "F1" if task == "binary" else "Macro-F1"
    head = f"| Model | ROC-AUC [95% CI] | Accuracy [95% CI] | {f1} [95% CI] |"
    extra = " Sensitivity | Specificity |" if task == "binary" else \
        " Recall normal | Recall bacterial | Recall viral |"
    lines = [head + extra + " ECE (before -> after) |",
             "|---|---|---|---|" + "---|" * (2 if task == "binary" else 3) + "---|"]
    for r in results:
        t, ci = r["test"], r["ci95"]
        cells = [f"{t[k]:.3f} [{ci[k][0]:.3f}-{ci[k][1]:.3f}]" for k in ["auc", "accuracy", "f1"]]
        if task == "binary":
            cells += [f"{t['sensitivity']:.3f}", f"{t['specificity']:.3f}"]
        else:
            cells += [f"{t[f'recall_{c}']:.3f}" for c in ["normal", "bacterial", "viral"]]
        cells.append(f"{r['ece_before']:.3f} -> {r['ece_after']:.3f}")
        name = f"**{r['pretty']}**" if r["name"] == "ensemble" else r["pretty"]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", f"Ensemble members (top-{len(ens_members)} by validation AUC): "
              + ", ".join(config.PRETTY[m] for m in ens_members), "",
              "| Model | Test AUC plain | Test AUC with TTA | Referral: accuracy all -> kept "
              f"(refer {config.REFERRAL_RATE:.0%}) | Errors caught by referral |", "|---|---|---|---|---|"]
    for r in results:
        rf = r["referral"]
        lines.append(f"| {r['pretty']} | {r['test_auc_plain']:.3f} | {r['test_auc_tta']:.3f} | "
                     f"{rf['accuracy_all']:.3f} -> {rf['accuracy_kept']:.3f} "
                     f"({rf['referred_fraction']:.1%} referred) | "
                     f"{rf['errors_caught']}/{rf['errors_total']} |")
    return "\n".join(lines) + "\n"


def main(task: str):
    pdir, fdir = config.preds_dir(task), config.figures_dir(task)
    fdir.mkdir(parents=True, exist_ok=True)
    preds = {}
    for n in config.MODEL_NAMES:
        f = pdir / f"{n}.npz"
        if f.exists():
            z = dict(np.load(f))
            z["val_unc"], z["test_unc"] = z["val_mc_std"], z["test_mc_std"]
            preds[n] = z
    if not preds:
        raise SystemExit("no predictions found - run src.predict first")

    results = [evaluate_model(n, d, task) for n, d in preds.items()]
    ranked = [r["name"] for r in sorted(results, key=lambda r: -r["val_auc"])]
    k = min(config.ENSEMBLE_TOP_K, len(ranked))
    ens = build_ensemble(preds, ranked, k)
    results.append(evaluate_model("ensemble", ens, task))
    y_te = ens["y_test"]
    best_single = ranked[0]

    # Figures
    fig_comparison(results, task, fdir / "model_comparison.png")
    fig_roc(results, y_te, task, fdir / "roc_curves.png")
    for n in [best_single, "ensemble"]:
        fig_confusion(next(r for r in results if r["name"] == n), y_te, task,
                      fdir / f"confusion_{n}.png")
    best_r = next(r for r in results if r["name"] == best_single)
    fig_reliability(best_r, preds[best_single], fdir / "reliability.png")
    curves = []
    for n, d, lbl in [("ensemble", ens, "Ensemble - member disagreement"),
                      (best_single, preds[best_single], f"{config.PRETTY[best_single]} - MC dropout")]:
        r = next(r for r in results if r["name"] == n)
        cov, acc = coverage_curve(y_te, r["_P_test"], d["test_unc"], r["threshold"])
        curves.append((lbl, cov, acc))
    rng = np.random.default_rng(0)
    cov, acc = coverage_curve(y_te, best_r["_P_test"], rng.random(len(y_te)), best_r["threshold"])
    curves.append(("Random referral (baseline)", cov, acc))
    fig_referral(curves, fdir / "referral_curve.png")

    # Tables
    md = to_markdown(results, task, ens["members"])
    (config.task_dir(task) / "results.md").write_text(md)
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    summary = {"task": task, "class_names": config.CLASS_NAMES[task], "n_test": int(len(y_te)),
               "best_single_model": best_single, "ensemble_members": ens["members"],
               "models": clean}
    (config.task_dir(task) / "results.json").write_text(json.dumps(summary, indent=2, default=float))
    print(md)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(config.CLASS_NAMES), default="binary")
    main(p.parse_args().task)
