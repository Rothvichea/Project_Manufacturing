import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune per-product confidence thresholds for GOOD/BAD decision")
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--min-infer-conf", type=float, default=0.001, help="Low conf to collect raw max score")
    p.add_argument("--objective", type=str, default="f1", choices=["f1", "balanced_accuracy", "recall"])
    p.add_argument("--min-recall", type=float, default=0.0, help="Optional recall floor for selected threshold")
    p.add_argument("--steps", type=int, default=101, help="Threshold grid size between 0 and 1")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("validation/eval/Validation_and_Evaluation/thresholds_per_product.json"),
    )
    p.add_argument(
        "--products",
        nargs="+",
        default=["bottle", "screw", "transistor", "grid"],
    )
    return p.parse_args()


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def sdiv(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def metrics_from_cm(cm: Dict[str, int]) -> Dict[str, float]:
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    precision = sdiv(tp, tp + fp)
    recall = sdiv(tp, tp + fn)
    specificity = sdiv(tn, tn + fp)
    f1 = sdiv(2 * precision * recall, precision + recall)
    acc = sdiv(tp + tn, tp + tn + fp + fn)
    bal_acc = 0.5 * (recall + specificity)
    return {
        "accuracy": acc,
        "precision_bad": precision,
        "recall_bad": recall,
        "f1_bad": f1,
        "specificity": specificity,
        "balanced_accuracy": bal_acc,
    }


def find_latest_weight(product: str) -> Path:
    root = Path("runs/detect/validation/runs")
    cands = sorted(root.glob(f"mvtec_{product}_binary_v*/weights/best.pt"))
    if not cands:
        raise FileNotFoundError(f"No weights found for product={product}")

    def vnum(p: Path) -> int:
        name = p.parts[-3]  # mvtec_xxx_binary_vN
        try:
            return int(name.split("_v")[-1])
        except Exception:
            return -1

    cands.sort(key=vnum)
    return cands[-1]


def collect_scores(model: YOLO, product: str, imgsz: int, device: str, min_infer_conf: float) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    root = Path("data/mvtec") / product / "test"
    items: List[Tuple[Path, int]] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        gt_bad = 0 if d.name == "good" else 1
        for img in sorted(d.glob("*.png")):
            items.append((img, gt_bad))

    y_true = np.zeros((len(items),), dtype=np.int32)
    y_score = np.zeros((len(items),), dtype=np.float32)
    paths: List[str] = []

    for i, (img, gt_bad) in enumerate(items):
        res = model.predict(
            source=str(img),
            conf=min_infer_conf,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )[0]
        if res.boxes is not None and len(res.boxes) > 0:
            score = float(np.max(res.boxes.conf.detach().cpu().numpy()))
        else:
            score = 0.0
        y_true[i] = gt_bad
        y_score[i] = score
        paths.append(str(img))

    return y_true, y_score, paths


def tune_threshold(y_true: np.ndarray, y_score: np.ndarray, steps: int, objective: str, min_recall: float) -> Dict[str, object]:
    thresholds = np.linspace(0.0, 1.0, steps)
    best = None
    best_val = -1.0
    all_points = []

    for t in thresholds:
        y_pred = (y_score >= t).astype(np.int32)
        cm = confusion(y_true, y_pred)
        m = metrics_from_cm(cm)
        if m["recall_bad"] < min_recall:
            continue

        score = m["f1_bad"] if objective == "f1" else (m["balanced_accuracy"] if objective == "balanced_accuracy" else m["recall_bad"])
        point = {"threshold": float(t), "metrics": m, "confusion": cm, "objective_value": float(score)}
        all_points.append(point)

        if score > best_val:
            best_val = score
            best = point

    if best is None:
        # fallback if recall constraint filtered all points
        t = 0.10
        y_pred = (y_score >= t).astype(np.int32)
        cm = confusion(y_true, y_pred)
        m = metrics_from_cm(cm)
        best = {"threshold": float(t), "metrics": m, "confusion": cm, "objective_value": float(m["f1_bad"])}

    return {"best": best, "curve": all_points}


def main() -> None:
    args = parse_args()

    out = {
        "settings": {
            "device": args.device,
            "imgsz": args.imgsz,
            "min_infer_conf": args.min_infer_conf,
            "objective": args.objective,
            "min_recall": args.min_recall,
            "steps": args.steps,
        },
        "products": {},
    }

    for product in args.products:
        w = find_latest_weight(product)
        model = YOLO(str(w))
        y_true, y_score, _ = collect_scores(model, product, args.imgsz, args.device, args.min_infer_conf)
        tune = tune_threshold(y_true, y_score, args.steps, args.objective, args.min_recall)

        out["products"][product] = {
            "weights": str(w),
            "n_images": int(len(y_true)),
            "n_bad": int(np.sum(y_true == 1)),
            "n_good": int(np.sum(y_true == 0)),
            "selected_threshold": tune["best"]["threshold"],
            "selected_metrics": tune["best"]["metrics"],
            "selected_confusion": tune["best"]["confusion"],
            "threshold_curve": tune["curve"],
        }

        sm = tune["best"]["metrics"]
        cm = tune["best"]["confusion"]
        print(
            f"[{product}] thr={tune['best']['threshold']:.3f} "
            f"Acc={sm['accuracy']:.4f} F1={sm['f1_bad']:.4f} Rec={sm['recall_bad']:.4f} "
            f"TP={cm['tp']} TN={cm['tn']} FP={cm['fp']} FN={cm['fn']}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
