import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test YOLO defect detector on labeled images")
    p.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to YOLO weights (.pt)",
    )
    p.add_argument(
        "--images",
        type=Path,
        default=Path("data/mvtec_yolo_defect/test/images"),
        help="Image directory",
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=Path("data/mvtec_yolo_defect/test/labels"),
        help="Label directory (YOLO txt)",
    )
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--device", type=str, default="0", help="CUDA device id or 'cpu'")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("validation/eval/mvtec_defect_test"),
        help="Output directory",
    )
    p.add_argument(
        "--max-examples-per-group",
        type=int,
        default=4,
        help="Max images per group (TP/TN/FP/FN) in examples figure",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Display the generated figure window",
    )
    return p.parse_args()


def list_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")
    images = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not images:
        raise RuntimeError(f"No images found in: {images_dir}")
    return sorted(images)


def has_defect_label(label_path: Path) -> bool:
    if not label_path.exists():
        return False
    text = label_path.read_text(encoding="utf-8").strip()
    return bool(text)


def confusion_counts(gt_bad: np.ndarray, pred_bad: np.ndarray) -> Dict[str, int]:
    tp = int(np.sum((gt_bad == 1) & (pred_bad == 1)))
    tn = int(np.sum((gt_bad == 0) & (pred_bad == 0)))
    fp = int(np.sum((gt_bad == 0) & (pred_bad == 1)))
    fn = int(np.sum((gt_bad == 1) & (pred_bad == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def draw_confusion(cm: Dict[str, int], save_path: Path) -> None:
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], dtype=np.int32)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mat, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred Good", "Pred Bad"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["GT Good", "GT Bad"])
    ax.set_title("Good/Bad Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center", color="black", fontsize=14)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def draw_examples(groups: Dict[str, List[np.ndarray]], save_path: Path, max_cols: int = 4, show: bool = False) -> None:
    order = ["tp", "tn", "fp", "fn"]
    titles = {"tp": "TP (Bad->Bad)", "tn": "TN (Good->Good)", "fp": "FP (Good->Bad)", "fn": "FN (Bad->Good)"}

    rows = len(order)
    cols = max_cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.1 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])

    for r, key in enumerate(order):
        imgs = groups.get(key, [])
        for c in range(cols):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(imgs):
                ax.imshow(cv2.cvtColor(imgs[c], cv2.COLOR_BGR2RGB))
            if c == 0:
                ax.set_title(titles[key], loc="left", fontsize=11, fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def draw_pred_good_bad(groups: Dict[str, List[np.ndarray]], save_path: Path, max_rows: int = 8, show: bool = False) -> None:
    cols = 2  # left=pred good, right=pred bad
    rows = max_rows
    fig, axes = plt.subplots(rows, cols, figsize=(9.0, 3.1 * rows))
    if rows == 1:
        axes = np.array([axes])

    pred_good = groups.get("pred_good", [])
    pred_bad = groups.get("pred_bad", [])

    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            ax.axis("off")
            if c == 0:
                if r < len(pred_good):
                    ax.imshow(cv2.cvtColor(pred_good[r], cv2.COLOR_BGR2RGB))
                if r == 0:
                    ax.set_title("Pred GOOD (No Defect Box)", fontsize=11, fontweight="bold")
            else:
                if r < len(pred_bad):
                    ax.imshow(cv2.cvtColor(pred_bad[r], cv2.COLOR_BGR2RGB))
                if r == 0:
                    ax.set_title("Pred BAD (Defect Box)", fontsize=11, fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    images = list_images(args.images)

    gt_list: List[int] = []
    pred_list: List[int] = []
    groups: Dict[str, List[np.ndarray]] = defaultdict(list)
    pred_groups: Dict[str, List[np.ndarray]] = defaultdict(list)

    # Stream inference and keep aligned with sorted image list
    results = model.predict(
        source=[str(p) for p in images],
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        stream=True,
        verbose=False,
        save=False,
    )

    for img_path, res in zip(images, results):
        label_path = args.labels / f"{img_path.stem}.txt"
        gt_bad = 1 if has_defect_label(label_path) else 0
        pred_bad = 1 if (res.boxes is not None and len(res.boxes) > 0) else 0

        gt_list.append(gt_bad)
        pred_list.append(pred_bad)

        if gt_bad == 1 and pred_bad == 1:
            key = "tp"
        elif gt_bad == 0 and pred_bad == 0:
            key = "tn"
        elif gt_bad == 0 and pred_bad == 1:
            key = "fp"
        else:
            key = "fn"

        if len(groups[key]) < args.max_examples_per_group:
            groups[key].append(res.plot())

        # Keep separate prediction-only gallery (good vs bad)
        pkey = "pred_bad" if pred_bad else "pred_good"
        if len(pred_groups[pkey]) < args.max_examples_per_group:
            pred_groups[pkey].append(res.plot())

    gt = np.array(gt_list, dtype=np.int32)
    pred = np.array(pred_list, dtype=np.int32)
    cm = confusion_counts(gt, pred)

    precision_bad = safe_div(cm["tp"], cm["tp"] + cm["fp"])
    recall_bad = safe_div(cm["tp"], cm["tp"] + cm["fn"])
    f1_bad = safe_div(2 * precision_bad * recall_bad, precision_bad + recall_bad)
    accuracy = safe_div(cm["tp"] + cm["tn"], len(gt))

    metrics = {
        "images": len(gt),
        "gt_bad": int(np.sum(gt == 1)),
        "gt_good": int(np.sum(gt == 0)),
        "pred_bad": int(np.sum(pred == 1)),
        "pred_good": int(np.sum(pred == 0)),
        "confusion": cm,
        "accuracy": accuracy,
        "precision_bad": precision_bad,
        "recall_bad": recall_bad,
        "f1_bad": f1_bad,
        "settings": {
            "weights": str(args.weights),
            "images": str(args.images),
            "labels": str(args.labels),
            "conf": args.conf,
            "imgsz": args.imgsz,
            "device": args.device,
        },
    }

    with (args.out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    summary = (
        f"Images: {metrics['images']}\n"
        f"GT bad/good: {metrics['gt_bad']}/{metrics['gt_good']}\n"
        f"Pred bad/good: {metrics['pred_bad']}/{metrics['pred_good']}\n"
        f"TP={cm['tp']} TN={cm['tn']} FP={cm['fp']} FN={cm['fn']}\n"
        f"Accuracy={accuracy:.4f} Precision_bad={precision_bad:.4f} "
        f"Recall_bad={recall_bad:.4f} F1_bad={f1_bad:.4f}\n"
    )
    (args.out / "summary.txt").write_text(summary, encoding="utf-8")

    draw_confusion(cm, args.out / "confusion_matrix_good_bad.png")
    draw_examples(
        groups=groups,
        save_path=args.out / "examples_tp_tn_fp_fn.png",
        max_cols=args.max_examples_per_group,
        show=args.show,
    )
    draw_pred_good_bad(
        groups=pred_groups,
        save_path=args.out / "examples_pred_good_vs_bad.png",
        max_rows=args.max_examples_per_group,
        show=False,
    )

    print(summary.strip())
    print(f"Saved: {args.out / 'metrics.json'}")
    print(f"Saved: {args.out / 'confusion_matrix_good_bad.png'}")
    print(f"Saved: {args.out / 'examples_tp_tn_fp_fn.png'}")
    print(f"Saved: {args.out / 'examples_pred_good_vs_bad.png'}")


if __name__ == "__main__":
    main()
