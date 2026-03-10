"""
Portfolio evaluation plots for Manufacturing Defect Detection project.
Reads fusion_metrics.json and fusion_predictions.csv from validation/results/
and writes all charts to validation/results/plots/.

Usage:
    python validation/plot_portfolio.py
"""

import json
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── paths ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent / "results"
METRICS_PATH = BASE / "fusion_metrics.json"
PREDS_PATH   = BASE / "fusion_predictions.csv"
OUT          = BASE / "plots"
OUT.mkdir(parents=True, exist_ok=True)

PRODUCTS = ["bottle", "screw", "transistor", "grid"]
COLORS   = {"bottle": "#4C72B0", "screw": "#DD8452", "transistor": "#55A868", "grid": "#C44E52"}
PALETTE  = list(COLORS.values())

# ── load data ──────────────────────────────────────────────────────────────
metrics = json.loads(METRICS_PATH.read_text())
with PREDS_PATH.open() as f:
    preds = list(csv.DictReader(f))

per = metrics["per_product"]
glob = metrics["global"]

# ── helpers ────────────────────────────────────────────────────────────────
def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  saved: {path}")
    plt.close(fig)


def bar_labels(ax, rects, fmt="{:.3f}", color="black", size=9):
    for r in rects:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width() / 2, h + 0.005, fmt.format(h),
                ha="center", va="bottom", fontsize=size, color=color, fontweight="bold")


# ══════════════════════════════════════════════════════════════════════════
# 1. GLOBAL SUMMARY CARD
# ══════════════════════════════════════════════════════════════════════════
def plot_global_summary():
    g = glob["metrics"]
    cm = glob["confusion"]
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    total = tp + tn + fp + fn

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor="#0f1117")
    fig.suptitle("Manufacturing Defect Detection — Global Performance",
                 fontsize=14, fontweight="bold", color="white", y=1.02)

    # ── metric bars ──
    ax = axes[0]
    ax.set_facecolor("#1a1d27")
    keys   = ["Accuracy", "Precision", "Recall", "F1"]
    vals   = [g["accuracy"], g["precision_bad"], g["recall_bad"], g["f1_bad"]]
    bar_c  = ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]
    rects  = ax.bar(keys, vals, color=bar_c, width=0.5, edgecolor="white", linewidth=0.4)
    bar_labels(ax, rects, color="white")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", color="white")
    ax.set_title("Overall Metrics", color="white", fontsize=11, pad=8)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.set_facecolor("#1a1d27")

    # ── confusion matrix ──
    ax = axes[1]
    ax.set_facecolor("#1a1d27")
    cm_arr = np.array([[tn, fp], [fn, tp]])
    im = ax.imshow(cm_arr, cmap="Blues", aspect="auto")
    labels = [["TN\n(Good→Good)", "FP\n(Good→BAD)"],
              ["FN\n(BAD→Good)", "TP\n(BAD→BAD)"]]
    thresh = cm_arr.max() / 2
    for i in range(2):
        for j in range(2):
            val = cm_arr[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{labels[i][j]}\n{val}", ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred GOOD", "Pred BAD"], color="white")
    ax.set_yticklabels(["GT GOOD", "GT BAD"], color="white")
    ax.set_title("Confusion Matrix", color="white", fontsize=11, pad=8)
    ax.tick_params(colors="white")

    # ── sample pie ──
    ax = axes[2]
    ax.set_facecolor("#1a1d27")
    sizes = [tp, tn, fp, fn]
    lbls  = [f"TP ({tp})", f"TN ({tn})", f"FP ({fp})", f"FN ({fn})"]
    clrs  = ["#55A868", "#4C72B0", "#DD8452", "#C44E52"]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=lbls, colors=clrs, autopct="%1.1f%%",
        startangle=90, textprops={"color": "white", "fontsize": 9},
        wedgeprops={"edgecolor": "#0f1117", "linewidth": 1.5}
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(8)
    ax.set_title(f"Prediction Breakdown\n(n={total})", color="white", fontsize=11, pad=8)

    fig.patch.set_facecolor("#0f1117")
    plt.tight_layout()
    save(fig, "01_global_summary.png")


# ══════════════════════════════════════════════════════════════════════════
# 2. PER-PRODUCT METRICS BAR CHART
# ══════════════════════════════════════════════════════════════════════════
def plot_per_product_metrics():
    metric_keys = ["accuracy", "precision_bad", "recall_bad", "f1_bad"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]
    n_metrics = len(metric_keys)
    n_products = len(PRODUCTS)
    x = np.arange(n_products)
    width = 0.18

    fig, ax = plt.subplots(figsize=(13, 5.5), facecolor="#0f1117")
    ax.set_facecolor("#1a1d27")

    hatch_styles = ["", "//", "xx", ".."]
    for i, (mk, ml) in enumerate(zip(metric_keys, metric_labels)):
        vals = [per[p]["metrics"][mk] for p in PRODUCTS]
        offset = (i - n_metrics / 2 + 0.5) * width
        rects = ax.bar(x + offset, vals, width, label=ml,
                       color=["#4C72B0", "#55A868", "#C44E52", "#DD8452"][i],
                       hatch=hatch_styles[i], edgecolor="white", linewidth=0.4, alpha=0.88)
        bar_labels(ax, rects, color="white", size=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([p.capitalize() for p in PRODUCTS], color="white", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", color="white", fontsize=11)
    ax.set_title("Per-Product Performance Metrics", color="white", fontsize=13, pad=10)
    ax.legend(loc="lower right", fontsize=9, facecolor="#2a2d37", labelcolor="white", edgecolor="#444")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.grid(axis="y", color="#333", linewidth=0.5, linestyle="--")
    fig.patch.set_facecolor("#0f1117")
    plt.tight_layout()
    save(fig, "02_per_product_metrics.png")


# ══════════════════════════════════════════════════════════════════════════
# 3. PER-PRODUCT CONFUSION MATRICES (2x2 grid)
# ══════════════════════════════════════════════════════════════════════════
def plot_per_product_confusion():
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), facecolor="#0f1117")
    fig.suptitle("Per-Product Confusion Matrices", color="white", fontsize=13, fontweight="bold", y=1.02)

    for ax, product in zip(axes, PRODUCTS):
        cm = per[product]["confusion"]
        m  = per[product]["metrics"]
        tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
        arr = np.array([[tn, fp], [fn, tp]])
        ax.set_facecolor("#1a1d27")
        im = ax.imshow(arr, cmap="Blues", aspect="auto", vmin=0, vmax=max(arr.max(), 1))
        cells = [[f"TN\n{tn}", f"FP\n{fp}"], [f"FN\n{fn}", f"TP\n{tp}"]]
        thresh = arr.max() / 2
        for i in range(2):
            for j in range(2):
                color = "white" if arr[i, j] > thresh else "black"
                ax.text(j, i, cells[i][j], ha="center", va="center",
                        fontsize=11, fontweight="bold", color=color)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Good", "Pred BAD"], color="white", fontsize=8)
        ax.set_yticklabels(["GT Good", "GT BAD"], color="white", fontsize=8)
        ax.tick_params(colors="white")
        f1 = m["f1_bad"]
        rec = m["recall_bad"]
        ax.set_title(f"{product.capitalize()}\nF1={f1:.3f}  Rec={rec:.3f}",
                     color=COLORS[product], fontsize=10, fontweight="bold", pad=6)
        ax.spines[:].set_color("#444")

    fig.patch.set_facecolor("#0f1117")
    plt.tight_layout()
    save(fig, "03_per_product_confusion.png")


# ══════════════════════════════════════════════════════════════════════════
# 4. SCORE DISTRIBUTIONS (YOLO / CNN / FUSED) PER PRODUCT
# ══════════════════════════════════════════════════════════════════════════
def plot_score_distributions():
    fig, axes = plt.subplots(4, 3, figsize=(14, 14), facecolor="#0f1117")
    fig.suptitle("Score Distributions: Good vs Bad (per product)",
                 color="white", fontsize=13, fontweight="bold", y=1.01)

    score_cols = [("yolo_score", "YOLO Raw Score"), ("cnn_prob", "CNN Prob Bad"), ("fused_score", "Fused Score")]
    bins = np.linspace(0, 1, 30)

    for row, product in enumerate(PRODUCTS):
        prod_rows = [r for r in preds if r["product"] == product]
        good = [r for r in prod_rows if r["gt_bad"] == "0"]
        bad  = [r for r in prod_rows if r["gt_bad"] == "1"]

        for col, (key, label) in enumerate(score_cols):
            ax = axes[row][col]
            ax.set_facecolor("#1a1d27")
            good_vals = [float(r[key]) for r in good]
            bad_vals  = [float(r[key]) for r in bad]
            ax.hist(good_vals, bins=bins, alpha=0.65, color="#4C72B0", label="Good", edgecolor="white", linewidth=0.3)
            ax.hist(bad_vals,  bins=bins, alpha=0.65, color="#C44E52", label="BAD",  edgecolor="white", linewidth=0.3)
            if col == 0:
                ax.set_ylabel(product.capitalize(), color=COLORS[product], fontsize=10, fontweight="bold")
            if row == 0:
                ax.set_title(label, color="white", fontsize=10, pad=6)
            ax.tick_params(colors="white", labelsize=7)
            ax.spines[:].set_color("#444")
            ax.legend(fontsize=7, facecolor="#2a2d37", labelcolor="white", edgecolor="#555")
            ax.grid(axis="y", color="#333", linewidth=0.4, linestyle="--")

    fig.patch.set_facecolor("#0f1117")
    plt.tight_layout()
    save(fig, "04_score_distributions.png")


# ══════════════════════════════════════════════════════════════════════════
# 5. TP / FP / FN / TN STACKED BAR (absolute counts)
# ══════════════════════════════════════════════════════════════════════════
def plot_stacked_outcomes():
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0f1117")
    ax.set_facecolor("#1a1d27")

    x      = np.arange(len(PRODUCTS))
    tp_v   = [per[p]["confusion"]["tp"] for p in PRODUCTS]
    tn_v   = [per[p]["confusion"]["tn"] for p in PRODUCTS]
    fp_v   = [per[p]["confusion"]["fp"] for p in PRODUCTS]
    fn_v   = [per[p]["confusion"]["fn"] for p in PRODUCTS]

    bars_tn = ax.bar(x, tn_v,                  label="TN (correct good)",  color="#4C72B0", edgecolor="white", linewidth=0.4)
    bars_tp = ax.bar(x, tp_v, bottom=tn_v,     label="TP (correct bad)",   color="#55A868", edgecolor="white", linewidth=0.4)
    fp_bot  = [tn_v[i] + tp_v[i] for i in range(len(x))]
    bars_fp = ax.bar(x, fp_v, bottom=fp_bot,   label="FP (false alarm)",   color="#DD8452", edgecolor="white", linewidth=0.4)
    fn_bot  = [fp_bot[i] + fp_v[i] for i in range(len(x))]
    bars_fn = ax.bar(x, fn_v, bottom=fn_bot,   label="FN (missed defect)", color="#C44E52", edgecolor="white", linewidth=0.4)

    # annotate FN (most critical)
    for i, (r, bot, v) in enumerate(zip(bars_fn, fn_bot, fn_v)):
        if v > 0:
            ax.text(r.get_x() + r.get_width() / 2, bot + v / 2,
                    f"FN={v}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    totals = [tn_v[i] + tp_v[i] + fp_v[i] + fn_v[i] for i in range(len(x))]
    for i, t in enumerate(totals):
        ax.text(x[i], t + 1, f"n={t}", ha="center", va="bottom", fontsize=9, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels([p.capitalize() for p in PRODUCTS], color="white", fontsize=11)
    ax.set_ylabel("Image Count", color="white", fontsize=11)
    ax.set_title("Prediction Outcomes per Product\n(FN = missed defect — critical in manufacturing)",
                 color="white", fontsize=12, pad=8)
    ax.legend(loc="upper right", fontsize=9, facecolor="#2a2d37", labelcolor="white", edgecolor="#444")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.grid(axis="y", color="#333", linewidth=0.5, linestyle="--")
    fig.patch.set_facecolor("#0f1117")
    plt.tight_layout()
    save(fig, "05_prediction_outcomes.png")


# ══════════════════════════════════════════════════════════════════════════
# 6. ROC-STYLE CURVE (fused_score threshold sweep per product)
# ══════════════════════════════════════════════════════════════════════════
def plot_roc_curves():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="#0f1117")

    thresholds = np.linspace(0, 1, 101)

    # Left: ROC (FPR vs TPR) for each product using fused_score
    ax = axes[0]
    ax.set_facecolor("#1a1d27")
    ax.plot([0, 1], [0, 1], "--", color="#555", linewidth=1, label="Random")

    for product in PRODUCTS:
        prod_rows = [r for r in preds if r["product"] == product]
        gt   = np.array([int(r["gt_bad"]) for r in prod_rows])
        score = np.array([float(r["fused_score"]) for r in prod_rows])
        tprs, fprs = [], []
        for thr in thresholds:
            pred = (score >= thr).astype(int)
            tp = int(((pred == 1) & (gt == 1)).sum())
            tn = int(((pred == 0) & (gt == 0)).sum())
            fp = int(((pred == 1) & (gt == 0)).sum())
            fn = int(((pred == 0) & (gt == 1)).sum())
            tprs.append(tp / max(tp + fn, 1))
            fprs.append(fp / max(fp + tn, 1))
        # auc (trapezoidal)
        fprs_arr = np.array(fprs[::-1])
        tprs_arr = np.array(tprs[::-1])
        auc = float(np.trapz(tprs_arr, fprs_arr))
        ax.plot(fprs, tprs, color=COLORS[product], linewidth=2,
                label=f"{product.capitalize()} (AUC={auc:.3f})")

    ax.set_xlabel("False Positive Rate", color="white", fontsize=10)
    ax.set_ylabel("True Positive Rate (Recall)", color="white", fontsize=10)
    ax.set_title("ROC Curve — Fused Score", color="white", fontsize=11, pad=8)
    ax.legend(fontsize=9, facecolor="#2a2d37", labelcolor="white", edgecolor="#444")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.grid(color="#333", linewidth=0.4, linestyle="--")

    # Right: Precision-Recall curve
    ax = axes[1]
    ax.set_facecolor("#1a1d27")
    for product in PRODUCTS:
        prod_rows = [r for r in preds if r["product"] == product]
        gt    = np.array([int(r["gt_bad"]) for r in prod_rows])
        score = np.array([float(r["fused_score"]) for r in prod_rows])
        precs, recs = [], []
        for thr in thresholds:
            pred = (score >= thr).astype(int)
            tp = int(((pred == 1) & (gt == 1)).sum())
            fp = int(((pred == 1) & (gt == 0)).sum())
            fn = int(((pred == 0) & (gt == 1)).sum())
            precs.append(tp / max(tp + fp, 1))
            recs.append(tp / max(tp + fn, 1))
        recs_arr  = np.array(recs[::-1])
        precs_arr = np.array(precs[::-1])
        auc_pr = float(np.trapz(precs_arr, recs_arr))
        ax.plot(recs, precs, color=COLORS[product], linewidth=2,
                label=f"{product.capitalize()} (AUC={auc_pr:.3f})")

    ax.set_xlabel("Recall", color="white", fontsize=10)
    ax.set_ylabel("Precision", color="white", fontsize=10)
    ax.set_title("Precision-Recall Curve — Fused Score", color="white", fontsize=11, pad=8)
    ax.legend(fontsize=9, facecolor="#2a2d37", labelcolor="white", edgecolor="#444")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.grid(color="#333", linewidth=0.4, linestyle="--")

    fig.patch.set_facecolor("#0f1117")
    plt.tight_layout()
    save(fig, "06_roc_pr_curves.png")


# ══════════════════════════════════════════════════════════════════════════
# 7. FUSION PARAMS USED (radar / spider chart)
# ══════════════════════════════════════════════════════════════════════════
def plot_fusion_params_radar():
    fusion_data = json.loads((BASE / "thresholds_per_product.json").read_text())
    categories  = ["alpha", "fused_threshold", "min_bad_score", "heatmap_threshold"]
    cat_labels  = ["Alpha\n(YOLO weight)", "Fused\nThreshold", "Min Bad\nScore", "Heatmap\nThreshold"]
    N = len(categories)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True}, facecolor="#0f1117")
    ax.set_facecolor("#1a1d27")

    for product in PRODUCTS:
        fp = fusion_data["products"][product].get("fusion_params", {})
        vals = [fp.get(c, 0) for c in categories]
        vals += vals[:1]
        ax.plot(angles, vals, color=COLORS[product], linewidth=2, label=product.capitalize())
        ax.fill(angles, vals, color=COLORS[product], alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cat_labels, color="white", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.tick_params(colors="white")
    ax.spines["polar"].set_color("#444")
    ax.grid(color="#444", linewidth=0.5)
    ax.set_title("Per-Product Fusion Parameters", color="white", fontsize=12, pad=20, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1),
              fontsize=9, facecolor="#2a2d37", labelcolor="white", edgecolor="#444")

    fig.patch.set_facecolor("#0f1117")
    save(fig, "07_fusion_params_radar.png")


# ══════════════════════════════════════════════════════════════════════════
# 8. COMBINED PORTFOLIO BANNER (single wide summary image)
# ══════════════════════════════════════════════════════════════════════════
def plot_portfolio_banner():
    g = glob["metrics"]
    cm = glob["confusion"]
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]

    fig = plt.figure(figsize=(16, 9), facecolor="#0f1117")
    gs  = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)

    # ── Title ──
    fig.text(0.5, 0.97,
             "Manufacturing Defect Detection — YOLO + VGG16 Fusion Pipeline",
             ha="center", va="top", fontsize=15, fontweight="bold", color="white")
    fig.text(0.5, 0.92,
             f"MVTec Dataset | 4 Products | 421 Test Images | F1={g['f1_bad']:.3f}  "
             f"Recall={g['recall_bad']:.3f}  Precision={g['precision_bad']:.3f}",
             ha="center", va="top", fontsize=10, color="#aaa")

    # ── Global metrics bar (top-left 2 cols) ──
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor("#1a1d27")
    keys  = ["Accuracy", "Precision", "Recall", "F1"]
    vals  = [g["accuracy"], g["precision_bad"], g["recall_bad"], g["f1_bad"]]
    clrs  = ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]
    rects = ax1.bar(keys, vals, color=clrs, width=0.5, edgecolor="white", linewidth=0.4)
    bar_labels(ax1, rects, color="white", size=9)
    ax1.set_ylim(0, 1.12)
    ax1.set_title("Global Metrics", color="white", fontsize=10)
    ax1.tick_params(colors="white")
    ax1.spines[:].set_color("#444")
    ax1.grid(axis="y", color="#333", linewidth=0.4, linestyle="--")

    # ── Global confusion (top-right 2 cols) ──
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.set_facecolor("#1a1d27")
    arr = np.array([[tn, fp], [fn, tp]])
    ax2.imshow(arr, cmap="Blues", aspect="auto")
    cells = [[f"TN\n{tn}", f"FP\n{fp}"], [f"FN\n{fn}", f"TP\n{tp}"]]
    thresh = arr.max() / 2
    for i in range(2):
        for j in range(2):
            color = "white" if arr[i, j] > thresh else "black"
            ax2.text(j, i, cells[i][j], ha="center", va="center",
                     fontsize=12, fontweight="bold", color=color)
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["Pred Good", "Pred BAD"], color="white")
    ax2.set_yticklabels(["GT Good", "GT BAD"], color="white")
    ax2.tick_params(colors="white")
    ax2.set_title("Confusion Matrix", color="white", fontsize=10)
    ax2.spines[:].set_color("#444")

    # ── Per-product F1 (bottom left) ──
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.set_facecolor("#1a1d27")
    f1_vals = [per[p]["metrics"]["f1_bad"] for p in PRODUCTS]
    rec_vals= [per[p]["metrics"]["recall_bad"] for p in PRODUCTS]
    x = np.arange(len(PRODUCTS))
    rects_f1  = ax3.bar(x - 0.2, f1_vals,  0.35, label="F1",    color=PALETTE, edgecolor="white", linewidth=0.4)
    rects_rec = ax3.bar(x + 0.2, rec_vals, 0.35, label="Recall", color=PALETTE, edgecolor="white", linewidth=0.4, alpha=0.5)
    bar_labels(ax3, rects_f1,  color="white", size=8)
    bar_labels(ax3, rects_rec, color="white", size=8)
    ax3.set_xticks(x)
    ax3.set_xticklabels([p.capitalize() for p in PRODUCTS], color="white", fontsize=9)
    ax3.set_ylim(0, 1.12)
    ax3.set_title("F1 and Recall per Product", color="white", fontsize=10)
    ax3.tick_params(colors="white")
    ax3.spines[:].set_color("#444")
    ax3.grid(axis="y", color="#333", linewidth=0.4, linestyle="--")
    f1_patch  = mpatches.Patch(color="#777", label="F1 (solid)")
    rec_patch = mpatches.Patch(color="#777", alpha=0.5, label="Recall (faded)")
    ax3.legend(handles=[f1_patch, rec_patch], fontsize=8, facecolor="#2a2d37",
               labelcolor="white", edgecolor="#444")

    # ── FN bar (bottom right) ──
    ax4 = fig.add_subplot(gs[1, 2:])
    ax4.set_facecolor("#1a1d27")
    fn_vals = [per[p]["confusion"]["fn"] for p in PRODUCTS]
    fp_vals = [per[p]["confusion"]["fp"] for p in PRODUCTS]
    x = np.arange(len(PRODUCTS))
    rects_fn = ax4.bar(x - 0.2, fn_vals, 0.35, label="FN (missed defect)", color="#C44E52", edgecolor="white", linewidth=0.4)
    rects_fp = ax4.bar(x + 0.2, fp_vals, 0.35, label="FP (false alarm)",   color="#DD8452", edgecolor="white", linewidth=0.4)
    bar_labels(ax4, rects_fn, fmt="{:.0f}", color="white", size=9)
    bar_labels(ax4, rects_fp, fmt="{:.0f}", color="white", size=9)
    ax4.set_xticks(x)
    ax4.set_xticklabels([p.capitalize() for p in PRODUCTS], color="white", fontsize=9)
    ax4.set_ylabel("Count", color="white", fontsize=9)
    ax4.set_title("Errors: FN (missed) vs FP (false alarm)", color="white", fontsize=10)
    ax4.tick_params(colors="white")
    ax4.spines[:].set_color("#444")
    ax4.legend(fontsize=8, facecolor="#2a2d37", labelcolor="white", edgecolor="#444")
    ax4.grid(axis="y", color="#333", linewidth=0.4, linestyle="--")

    fig.patch.set_facecolor("#0f1117")
    save(fig, "00_portfolio_banner.png")


# ══════════════════════════════════════════════════════════════════════════
# 8. GLOBAL ACCURACY DASHBOARD — gauge + before/after improvement
# ══════════════════════════════════════════════════════════════════════════
def plot_global_accuracy_dashboard():
    g  = glob["metrics"]
    cm = glob["confusion"]

    # before-fix values (from original heatmap_priority run)
    before = {"accuracy": 0.9264, "precision_bad": 0.9921, "recall_bad": 0.8961, "f1_bad": 0.9416,
              "fn": 29, "fp": 2}
    after  = {"accuracy": g["accuracy"], "precision_bad": g["precision_bad"],
              "recall_bad": g["recall_bad"], "f1_bad": g["f1_bad"],
              "fn": cm["fn"], "fp": cm["fp"]}

    fig = plt.figure(figsize=(16, 8), facecolor="#0f1117")
    fig.suptitle("Global Performance Dashboard — YOLO+CNN Fusion Pipeline",
                 color="white", fontsize=14, fontweight="bold", y=1.01)

    gs = GridSpec(2, 4, figure=fig, hspace=0.55, wspace=0.42)

    # ── gauge-style arc for each metric (top row) ──
    metric_defs = [
        ("Accuracy",  g["accuracy"],       "#4C72B0",
         "Correct predictions\nout of all images"),
        ("Precision", g["precision_bad"],  "#55A868",
         "BAD alarms that are\ntruly defective"),
        ("Recall",    g["recall_bad"],     "#C44E52",
         "Defects caught\nout of all defects"),
        ("F1 Score",  g["f1_bad"],         "#DD8452",
         "Harmonic mean of\nPrecision & Recall"),
    ]
    for col, (label, val, color, desc) in enumerate(metric_defs):
        ax = fig.add_subplot(gs[0, col], polar=True)
        ax.set_facecolor("#1a1d27")

        # background arc (full 0→1, grey)
        theta_bg  = np.linspace(np.pi, 0, 200)
        ax.plot(theta_bg, [1]*200, color="#333", linewidth=14, solid_capstyle="round")

        # value arc (0→val, coloured)
        theta_val = np.linspace(np.pi, np.pi - val * np.pi, 200)
        ax.plot(theta_val, [1]*200, color=color, linewidth=14, solid_capstyle="round")

        # "0%" label at left end of arc
        ax.text(np.pi + 0.08, 1.0, "0%", ha="left", va="center",
                fontsize=7, color="#888", transform=ax.transData)
        # "100%" label at right end of arc
        ax.text(-0.08, 1.0, "100%", ha="right", va="center",
                fontsize=7, color="#888", transform=ax.transData)

        # big numeric value in centre
        ax.text(0, 0.18, f"{val*100:.2f}%", ha="center", va="center",
                fontsize=17, fontweight="bold", color=color,
                transform=ax.transData)

        # metric name just below the value
        ax.text(0, -0.15, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white",
                transform=ax.transData)

        # short description even further below
        ax.text(0, -0.52, desc, ha="center", va="center",
                fontsize=7.5, color="#aaa", linespacing=1.4,
                transform=ax.transData)

        ax.set_ylim(0, 1.35)
        ax.set_xlim(0, np.pi)
        ax.axis("off")
        ax.set_facecolor("#1a1d27")
        fig.patch.set_facecolor("#0f1117")

    # ── before vs after: key metrics (bottom left 2 cols) ──
    ax_cmp = fig.add_subplot(gs[1, :2])
    ax_cmp.set_facecolor("#1a1d27")
    keys   = ["Accuracy", "Recall", "F1"]
    b_vals = [before["accuracy"], before["recall_bad"], before["f1_bad"]]
    a_vals = [after["accuracy"],  after["recall_bad"],  after["f1_bad"]]
    x      = np.arange(len(keys))
    rb = ax_cmp.bar(x - 0.2, b_vals, 0.35, label="Before fix", color="#666",
                    edgecolor="white", linewidth=0.4)
    ra = ax_cmp.bar(x + 0.2, a_vals, 0.35, label="After fix",  color="#55A868",
                    edgecolor="white", linewidth=0.4)
    for rect, v in zip(list(rb)+list(ra), b_vals+a_vals):
        ax_cmp.text(rect.get_x()+rect.get_width()/2, rect.get_height()+0.002,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8,
                    color="white", fontweight="bold")
    ax_cmp.set_xticks(x)
    ax_cmp.set_xticklabels(keys, color="white", fontsize=10)
    ax_cmp.set_ylim(0.85, 1.02)
    ax_cmp.set_title("Before vs. After Per-Product Threshold Fix",
                     color="white", fontsize=10, pad=6)
    ax_cmp.legend(fontsize=9, facecolor="#2a2d37", labelcolor="white", edgecolor="#444")
    ax_cmp.tick_params(colors="white")
    ax_cmp.spines[:].set_color("#444")
    ax_cmp.grid(axis="y", color="#333", linewidth=0.4, linestyle="--")

    # ── FN reduction bar (bottom right 2 cols) ──
    ax_fn = fig.add_subplot(gs[1, 2:])
    ax_fn.set_facecolor("#1a1d27")
    prod_labels = ["Bottle", "Screw", "Transistor", "Grid", "Global"]
    fn_before = [1, 21, 7, 0, 29]
    fn_after  = [per[p]["confusion"]["fn"] for p in PRODUCTS] + [cm["fn"]]
    x2 = np.arange(len(prod_labels))
    rb2 = ax_fn.bar(x2 - 0.2, fn_before, 0.35, label="FN before", color="#C44E52",
                    edgecolor="white", linewidth=0.4)
    ra2 = ax_fn.bar(x2 + 0.2, fn_after,  0.35, label="FN after",  color="#55A868",
                    edgecolor="white", linewidth=0.4)
    for rect, v in zip(list(rb2)+list(ra2), fn_before+fn_after):
        ax_fn.text(rect.get_x()+rect.get_width()/2, rect.get_height()+0.1,
                   str(v), ha="center", va="bottom", fontsize=9,
                   color="white", fontweight="bold")
    ax_fn.set_xticks(x2)
    ax_fn.set_xticklabels(prod_labels, color="white", fontsize=9)
    ax_fn.set_ylabel("Missed Defects (FN)", color="white", fontsize=9)
    ax_fn.set_title("False Negatives (Missed Defects) Before vs. After Fix",
                    color="white", fontsize=10, pad=6)
    ax_fn.legend(fontsize=9, facecolor="#2a2d37", labelcolor="white", edgecolor="#444")
    ax_fn.tick_params(colors="white")
    ax_fn.spines[:].set_color("#444")
    ax_fn.grid(axis="y", color="#333", linewidth=0.4, linestyle="--")

    fig.patch.set_facecolor("#0f1117")
    plt.tight_layout()
    save(fig, "08_global_accuracy_dashboard.png")


# ── run all ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating portfolio plots...")
    plot_portfolio_banner()
    plot_global_summary()
    plot_per_product_metrics()
    plot_per_product_confusion()
    plot_score_distributions()
    plot_stacked_outcomes()
    plot_roc_curves()
    plot_fusion_params_radar()
    plot_global_accuracy_dashboard()
    print(f"\nAll plots saved to: {OUT}")
