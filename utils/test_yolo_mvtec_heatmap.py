import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate YOLO on raw MVTec with GT/pred heatmaps")
    p.add_argument("--weights", type=Path, required=True, help="Path to YOLO weights")
    p.add_argument("--mvtec-root", type=Path, default=Path("data/mvtec"), help="MVTec root")
    p.add_argument(
        "--categories",
        nargs="+",
        default=["bottle", "screw", "transistor", "grid"],
        help="MVTec categories to evaluate",
    )
    p.add_argument("--conf", type=float, default=0.10, help="YOLO confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--device", type=str, default="0", help="CUDA device id or 'cpu'")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("validation/eval/Validation_and_Evaluation"),
        help="Output directory",
    )
    p.add_argument(
        "--max-save-per-split",
        type=int,
        default=120,
        help="Max images saved per category/pred split",
    )
    return p.parse_args()


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def confusion_to_metrics(cm: Dict[str, int]) -> Dict[str, float]:
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    acc = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "accuracy": acc,
        "precision_bad": precision,
        "recall_bad": recall,
        "f1_bad": f1,
        "specificity": safe_div(tn, tn + fp),
        "balanced_accuracy": 0.5 * (recall + safe_div(tn, tn + fp)),
        "mcc": safe_div((tp * tn - fp * fn), np.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))),
    }


def init_cm() -> Dict[str, int]:
    return {"tp": 0, "tn": 0, "fp": 0, "fn": 0}


def update_cm(cm: Dict[str, int], gt_bad: int, pred_bad: int) -> None:
    if gt_bad == 1 and pred_bad == 1:
        cm["tp"] += 1
    elif gt_bad == 0 and pred_bad == 0:
        cm["tn"] += 1
    elif gt_bad == 0 and pred_bad == 1:
        cm["fp"] += 1
    else:
        cm["fn"] += 1


def draw_confusion(cm: Dict[str, int], save_path: Path, title: str) -> None:
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], dtype=np.int32)
    cell = 170
    left = 220
    top = 110
    w = left + 2 * cell + 30
    h = top + 2 * cell + 40
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    vmax = int(np.max(mat)) if int(np.max(mat)) > 0 else 1
    for r in range(2):
        for c in range(2):
            val = int(mat[r, c])
            intensity = int(255 - 210 * (val / vmax))
            color = (255, intensity, intensity)  # BGR
            x1 = left + c * cell
            y1 = top + r * cell
            x2 = x1 + cell
            y2 = y1 + cell
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 60, 60), 2)
            cv2.putText(
                canvas,
                str(val),
                (x1 + cell // 2 - 22, y1 + cell // 2 + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 0),
                2,
            )

    cv2.putText(canvas, title, (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(canvas, "Pred Good", (left + 20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, "Pred Bad", (left + cell + 25, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, "GT Good", (40, top + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, "GT Bad", (52, top + cell + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imwrite(str(save_path), canvas)


def compute_roc_pr(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, np.ndarray]:
    thresholds = np.linspace(1.0, 0.0, 201)
    roc_fpr, roc_tpr = [], []
    pr_recall, pr_precision = [], []

    for t in thresholds:
        y_pred = (y_score >= t).astype(np.int32)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        tpr = safe_div(tp, tp + fn)
        fpr = safe_div(fp, fp + tn)
        precision = safe_div(tp, tp + fp)
        recall = tpr

        roc_fpr.append(fpr)
        roc_tpr.append(tpr)
        pr_recall.append(recall)
        pr_precision.append(precision)

    roc_fpr = np.array(roc_fpr, dtype=np.float32)
    roc_tpr = np.array(roc_tpr, dtype=np.float32)
    pr_recall = np.array(pr_recall, dtype=np.float32)
    pr_precision = np.array(pr_precision, dtype=np.float32)

    roc_order = np.argsort(roc_fpr)
    pr_order = np.argsort(pr_recall)

    auc_roc = float(np.trapz(roc_tpr[roc_order], roc_fpr[roc_order]))
    auc_pr = float(np.trapz(pr_precision[pr_order], pr_recall[pr_order]))

    return {
        "thresholds": thresholds.astype(np.float32),
        "roc_fpr": roc_fpr,
        "roc_tpr": roc_tpr,
        "pr_recall": pr_recall,
        "pr_precision": pr_precision,
        "auc_roc": np.array([auc_roc], dtype=np.float32),
        "auc_pr": np.array([auc_pr], dtype=np.float32),
    }


def _blank_plot(title: str) -> Tuple[np.ndarray, int, int, int, int]:
    w, h = 920, 650
    left, top, right, bottom = 100, 70, 70, 90
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (left, top), (w - right, h - bottom), (40, 40, 40), 2)
    cv2.putText(img, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    return img, left, top, w - right, h - bottom


def _map_xy(x: float, y: float, left: int, top: int, right: int, bottom: int) -> Tuple[int, int]:
    px = int(left + x * (right - left))
    py = int(bottom - y * (bottom - top))
    return px, py


def draw_curve_plot(
    x: np.ndarray,
    y: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    save_path: Path,
    diagonal: bool = False,
    auc_text: str = "",
) -> None:
    img, left, top, right, bottom = _blank_plot(title)

    cv2.putText(img, x_label, ((left + right) // 2 - 40, bottom + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, y_label, (15, (top + bottom) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    if diagonal:
        p1 = _map_xy(0.0, 0.0, left, top, right, bottom)
        p2 = _map_xy(1.0, 1.0, left, top, right, bottom)
        cv2.line(img, p1, p2, (170, 170, 170), 2, cv2.LINE_AA)

    if len(x) > 1:
        pts = []
        for xv, yv in zip(x, y):
            xv = float(np.clip(xv, 0.0, 1.0))
            yv = float(np.clip(yv, 0.0, 1.0))
            pts.append(_map_xy(xv, yv, left, top, right, bottom))
        for i in range(1, len(pts)):
            cv2.line(img, pts[i - 1], pts[i], (30, 70, 220), 2, cv2.LINE_AA)

    if auc_text:
        cv2.putText(img, auc_text, (right - 260, top + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    for tick in range(6):
        v = tick / 5.0
        px, py = _map_xy(v, 0.0, left, top, right, bottom)
        cv2.putText(img, f"{v:.1f}", (px - 10, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        px2, py2 = _map_xy(0.0, v, left, top, right, bottom)
        cv2.putText(img, f"{v:.1f}", (left - 45, py2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    cv2.imwrite(str(save_path), img)


def draw_score_histogram(y_true: np.ndarray, y_score: np.ndarray, save_path: Path) -> None:
    img, left, top, right, bottom = _blank_plot("Score Distribution (Defect Confidence)")
    bins = np.linspace(0.0, 1.0, 21)

    good_scores = y_score[y_true == 0]
    bad_scores = y_score[y_true == 1]
    h_good, _ = np.histogram(good_scores, bins=bins)
    h_bad, _ = np.histogram(bad_scores, bins=bins)
    hmax = max(int(np.max(h_good)), int(np.max(h_bad)), 1)

    n_bins = len(bins) - 1
    width = (right - left) / n_bins
    for i in range(n_bins):
        x1 = int(left + i * width)
        x2 = int(left + (i + 1) * width) - 1

        gh = int((h_good[i] / hmax) * (bottom - top))
        bh = int((h_bad[i] / hmax) * (bottom - top))

        cv2.rectangle(img, (x1, bottom - gh), (x2, bottom), (80, 200, 80), -1)
        cv2.rectangle(img, (x1, bottom - bh), (x2, bottom - max(gh - 2, 0)), (60, 70, 230), -1)

    cv2.putText(img, "Green: GT Good", (left + 10, top + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 120, 40), 2)
    cv2.putText(img, "Blue: GT Bad", (left + 230, top + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 40, 30), 2)
    cv2.putText(img, "Confidence", ((left + right) // 2 - 45, bottom + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "Count", (20, (top + bottom) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.imwrite(str(save_path), img)


def draw_per_category_bars(per_cat_metrics: Dict[str, Dict[str, float]], save_path: Path) -> None:
    cats = list(per_cat_metrics.keys())
    if not cats:
        return

    img, left, top, right, bottom = _blank_plot("Per-Category Metrics")
    bar_w = 28
    group_w = 120
    start_x = left + 20
    scale_h = (bottom - top) - 40

    for i, cat in enumerate(cats):
        m = per_cat_metrics[cat]
        vals = [
            ("P", m["precision_bad"], (60, 70, 230)),
            ("R", m["recall_bad"], (80, 200, 80)),
            ("F1", m["f1_bad"], (230, 140, 60)),
            ("Acc", m["accuracy"], (150, 90, 180)),
        ]
        gx = start_x + i * group_w
        for j, (label, v, color) in enumerate(vals):
            x1 = gx + j * (bar_w + 2)
            x2 = x1 + bar_w
            h = int(np.clip(v, 0.0, 1.0) * scale_h)
            cv2.rectangle(img, (x1, bottom - h), (x2, bottom), color, -1)
            cv2.putText(img, label, (x1 + 2, bottom + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 0), 1)
        cv2.putText(img, cat[:10], (gx, bottom + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    cv2.putText(img, "Value [0..1]", (15, (top + bottom) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    cv2.imwrite(str(save_path), img)


def make_gt_mask(mask_path: Path, shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    if not mask_path.exists():
        return np.zeros((h, w), dtype=np.uint8)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros((h, w), dtype=np.uint8)
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def build_pred_heatmap(shape: Tuple[int, int], boxes_xyxy: np.ndarray, confs: np.ndarray) -> np.ndarray:
    h, w = shape
    heat = np.zeros((h, w), dtype=np.float32)
    if boxes_xyxy.size == 0:
        return heat

    for (x1, y1, x2, y2), conf in zip(boxes_xyxy, confs):
        ix1 = int(np.clip(np.floor(x1), 0, w - 1))
        iy1 = int(np.clip(np.floor(y1), 0, h - 1))
        ix2 = int(np.clip(np.ceil(x2), 0, w - 1))
        iy2 = int(np.clip(np.ceil(y2), 0, h - 1))
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        heat[iy1:iy2, ix1:ix2] += float(conf)

    if np.max(heat) > 0:
        heat = heat / (np.max(heat) + 1e-8)
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=9, sigmaY=9)
    if np.max(heat) > 0:
        heat = heat / (np.max(heat) + 1e-8)
    return heat


def overlay_heatmap(img_bgr: np.ndarray, heat: np.ndarray) -> np.ndarray:
    hmap_u8 = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(hmap_u8, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(img_bgr, 0.55, colored, 0.45, 0)
    return blended


def make_panel(
    original: np.ndarray,
    pred_vis: np.ndarray,
    gt_overlay: np.ndarray,
    pred_overlay: np.ndarray,
    top_text: str,
) -> np.ndarray:
    h, w = original.shape[:2]

    def add_title(img: np.ndarray, title: str) -> np.ndarray:
        canvas = np.full((h + 36, w, 3), 255, dtype=np.uint8)
        canvas[36:, :, :] = img
        cv2.putText(canvas, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        return canvas

    a = add_title(original, "Original")
    b = add_title(pred_vis, "YOLO Boxes")
    c = add_title(gt_overlay, "GT Heatmap")
    d = add_title(pred_overlay, "Pred Heatmap")

    row = np.hstack([a, b, c, d])
    header = np.full((40, row.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(header, top_text, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 2)
    return np.vstack([header, row])


def collect_mvtec_images(cat_root: Path) -> List[Tuple[Path, str]]:
    out: List[Tuple[Path, str]] = []
    test_root = cat_root / "test"
    for defect_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
        label = defect_dir.name  # good or defect subtype
        for img in sorted(defect_dir.glob("*.png")):
            out.append((img, label))
    return out


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))

    global_cm = init_cm()
    per_cat: Dict[str, Dict[str, int]] = {}
    rows: List[Dict[str, object]] = []
    all_gt: List[int] = []
    all_score: List[float] = []

    for cat in args.categories:
        cat_root = args.mvtec_root / cat
        if not cat_root.exists():
            print(f"[WARN] category not found: {cat}")
            continue

        per_cat[cat] = init_cm()
        saved_counts = {"pred_good": 0, "pred_bad": 0}

        img_items = collect_mvtec_images(cat_root)
        if not img_items:
            print(f"[WARN] no test images in: {cat}")
            continue

        cat_out = args.out / cat
        (cat_out / "pred_good").mkdir(parents=True, exist_ok=True)
        (cat_out / "pred_bad").mkdir(parents=True, exist_ok=True)

        for img_path, defect_label in img_items:
            gt_bad = 0 if defect_label == "good" else 1

            res = model.predict(
                source=str(img_path),
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )[0]

            pred_bad = 1 if (res.boxes is not None and len(res.boxes) > 0) else 0
            update_cm(per_cat[cat], gt_bad, pred_bad)
            update_cm(global_cm, gt_bad, pred_bad)

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            # Ground-truth mask for defect classes
            if gt_bad == 1:
                mask_path = cat_root / "ground_truth" / defect_label / f"{img_path.stem}_mask.png"
            else:
                mask_path = Path("__missing__")
            gt_mask = make_gt_mask(mask_path, (h, w))
            gt_heat = gt_mask.astype(np.float32)
            if np.max(gt_heat) > 0:
                gt_heat = gt_heat / 255.0

            # Pred heatmap from predicted boxes/confidences
            if res.boxes is not None and len(res.boxes) > 0:
                boxes_xyxy = res.boxes.xyxy.detach().cpu().numpy()
                confs = res.boxes.conf.detach().cpu().numpy()
            else:
                boxes_xyxy = np.zeros((0, 4), dtype=np.float32)
                confs = np.zeros((0,), dtype=np.float32)
            max_conf = float(np.max(confs)) if len(confs) else 0.0
            all_gt.append(gt_bad)
            all_score.append(max_conf)
            pred_heat = build_pred_heatmap((h, w), boxes_xyxy, confs)

            pred_vis = res.plot()
            gt_overlay = overlay_heatmap(img, gt_heat)
            pred_overlay = overlay_heatmap(img, pred_heat)

            title = (
                f"cat={cat} src={defect_label} "
                f"GT={'BAD' if gt_bad else 'GOOD'} PRED={'BAD' if pred_bad else 'GOOD'}"
            )
            panel = make_panel(img, pred_vis, gt_overlay, pred_overlay, title)

            split_name = "pred_bad" if pred_bad else "pred_good"
            if saved_counts[split_name] < args.max_save_per_split:
                out_name = f"{img_path.stem}_{defect_label}.png"
                cv2.imwrite(str(cat_out / split_name / out_name), panel)
                saved_counts[split_name] += 1

            rows.append(
                {
                    "category": cat,
                    "image": str(img_path),
                    "source_label": defect_label,
                    "gt_bad": gt_bad,
                    "pred_bad": pred_bad,
                    "pred_score": max_conf,
                    "num_boxes": int(len(boxes_xyxy)),
                }
            )

        cat_metrics = confusion_to_metrics(per_cat[cat])
        draw_confusion(
            per_cat[cat],
            cat_out / "confusion_matrix_good_bad.png",
            title=f"{cat}: Good/Bad Confusion",
        )
        print(
            f"[{cat}] TP={per_cat[cat]['tp']} TN={per_cat[cat]['tn']} "
            f"FP={per_cat[cat]['fp']} FN={per_cat[cat]['fn']} "
            f"Acc={cat_metrics['accuracy']:.4f} Rec_bad={cat_metrics['recall_bad']:.4f}"
        )

    # Global outputs
    global_metrics = confusion_to_metrics(global_cm)
    draw_confusion(global_cm, args.out / "confusion_matrix_good_bad_global.png", "Global Good/Bad Confusion")
    y_true = np.array(all_gt, dtype=np.int32)
    y_score = np.array(all_score, dtype=np.float32)
    curves = compute_roc_pr(y_true, y_score)
    auc_roc = float(curves["auc_roc"][0])
    auc_pr = float(curves["auc_pr"][0])

    roc_order = np.argsort(curves["roc_fpr"])
    pr_order = np.argsort(curves["pr_recall"])
    draw_curve_plot(
        curves["roc_fpr"][roc_order],
        curves["roc_tpr"][roc_order],
        x_label="False Positive Rate",
        y_label="True Positive Rate",
        title="ROC Curve",
        save_path=args.out / "roc_curve.png",
        diagonal=True,
        auc_text=f"AUC-ROC={auc_roc:.4f}",
    )
    draw_curve_plot(
        curves["pr_recall"][pr_order],
        curves["pr_precision"][pr_order],
        x_label="Recall",
        y_label="Precision",
        title="Precision-Recall Curve",
        save_path=args.out / "precision_recall_curve.png",
        diagonal=False,
        auc_text=f"AUC-PR={auc_pr:.4f}",
    )
    draw_score_histogram(y_true, y_score, args.out / "score_distribution_histogram.png")
    per_cat_metrics = {cat: confusion_to_metrics(per_cat[cat]) for cat in per_cat}
    draw_per_category_bars(per_cat_metrics, args.out / "per_category_metrics_bar.png")

    report = {
        "settings": {
            "weights": str(args.weights),
            "mvtec_root": str(args.mvtec_root),
            "categories": args.categories,
            "conf": args.conf,
            "imgsz": args.imgsz,
            "device": args.device,
        },
        "global_confusion": global_cm,
        "global_metrics": {
            **global_metrics,
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
        },
        "per_category": {
            cat: {
                "confusion": per_cat[cat],
                "metrics": per_cat_metrics[cat],
            }
            for cat in per_cat
        },
    }

    with (args.out / "metrics_by_category.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with (args.out / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["category", "image", "source_label", "gt_bad", "pred_bad", "pred_score", "num_boxes"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = (
        f"Global: TP={global_cm['tp']} TN={global_cm['tn']} FP={global_cm['fp']} FN={global_cm['fn']}\n"
        f"Accuracy={global_metrics['accuracy']:.4f} "
        f"Precision_bad={global_metrics['precision_bad']:.4f} "
        f"Recall_bad={global_metrics['recall_bad']:.4f} F1_bad={global_metrics['f1_bad']:.4f} "
        f"AUC-ROC={auc_roc:.4f} AUC-PR={auc_pr:.4f}\n"
    )
    (args.out / "summary.txt").write_text(summary, encoding="utf-8")

    print(summary.strip())
    print(f"Saved: {args.out / 'metrics_by_category.json'}")
    print(f"Saved: {args.out / 'predictions.csv'}")
    print(f"Saved: {args.out / 'confusion_matrix_good_bad_global.png'}")
    print(f"Saved: {args.out / 'roc_curve.png'}")
    print(f"Saved: {args.out / 'precision_recall_curve.png'}")
    print(f"Saved: {args.out / 'score_distribution_histogram.png'}")
    print(f"Saved: {args.out / 'per_category_metrics_bar.png'}")


if __name__ == "__main__":
    main()
