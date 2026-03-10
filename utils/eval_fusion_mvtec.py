import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate fused YOLO+CNN on raw MVTec test sets")
    p.add_argument("--products", nargs="+", default=["screw", "transistor"])
    p.add_argument("--mvtec-root", type=Path, default=Path("data/mvtec"))
    p.add_argument(
        "--threshold-config",
        type=Path,
        default=Path("validation/eval/Validation_and_Evaluation/thresholds_per_product.json"),
    )
    p.add_argument(
        "--cnn-weights",
        type=Path,
        default=Path("validation/runs/vgg16_cls_merged_v1/best.pt"),
    )
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--fused-threshold", type=float, default=0.5)
    p.add_argument(
        "--decision-mode",
        type=str,
        default="fused",
        choices=["fused", "heatmap_only", "heatmap_priority"],
        help=(
            "fused: use YOLO+CNN fused score; "
            "heatmap_only: use only pred heatmap threshold; "
            "heatmap_priority: if heatmap triggers BAD, override fused decision."
        ),
    )
    p.add_argument(
        "--heatmap-threshold",
        type=float,
        default=0.02,
        help="If max predicted heatmap >= threshold, heatmap is considered abnormal.",
    )
    p.add_argument(
        "--min-bad-score",
        type=float,
        default=0.20,
        help="If YOLO max score is below this value, force prediction to GOOD and hide boxes.",
    )
    p.add_argument("--max-images-per-product", type=int, default=0, help="0 means all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("validation/eval/Validation_and_Evaluation/fusion_mvtec_eval"),
    )
    return p.parse_args()


def resolve_device(dev_arg: str) -> torch.device:
    dev = str(dev_arg).strip().lower()
    if dev == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        if dev.isdigit():
            return torch.device(f"cuda:{dev}")
        if dev.startswith("cuda"):
            return torch.device(dev)
        return torch.device("cuda:0")
    return torch.device("cpu")


def sdiv(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def cm_metrics(cm: Dict[str, int]) -> Dict[str, float]:
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    p = sdiv(tp, tp + fp)
    r = sdiv(tp, tp + fn)
    f1 = sdiv(2 * p * r, p + r)
    acc = sdiv(tp + tn, tp + tn + fp + fn)
    return {"accuracy": acc, "precision_bad": p, "recall_bad": r, "f1_bad": f1}


def add_cm(cm: Dict[str, int], gt_bad: int, pred_bad: int) -> None:
    if gt_bad == 1 and pred_bad == 1:
        cm["tp"] += 1
    elif gt_bad == 0 and pred_bad == 0:
        cm["tn"] += 1
    elif gt_bad == 0 and pred_bad == 1:
        cm["fp"] += 1
    else:
        cm["fn"] += 1


def build_cnn(ckpt: Dict, device: torch.device) -> nn.Module:
    model = models.vgg16(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(25088, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(128, 32),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(32, 1),
    )
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    model.eval()
    return model


def cnn_prob_bad(img_bgr: np.ndarray, cnn: nn.Module, imgsz: int, device: torch.device) -> float:
    tf = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((imgsz, imgsz)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    x = tf(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = cnn(x).squeeze(1)
        return float(torch.sigmoid(logit).item())


def load_thresholds(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_gt_mask(mask_path: Path, shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    if not mask_path.exists():
        return np.zeros((h, w), dtype=np.uint8)
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros((h, w), dtype=np.uint8)
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return m


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
    hm = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    cm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 0.55, cm, 0.45, 0)


def make_panel(original: np.ndarray, yolo_vis: np.ndarray, gt_overlay: np.ndarray, pred_overlay: np.ndarray, text: str) -> np.ndarray:
    h, w = original.shape[:2]

    def tile(im: np.ndarray, title: str) -> np.ndarray:
        c = np.full((h + 34, w, 3), 255, dtype=np.uint8)
        c[34:, :, :] = im
        cv2.putText(c, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)
        return c

    a = tile(original, "Original")
    b = tile(yolo_vis, "YOLO/Fusion View")
    c = tile(gt_overlay, "GT Heatmap")
    d = tile(pred_overlay, "Pred Heatmap")
    row = np.hstack([a, b, c, d])
    header = np.full((44, row.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(header, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)
    return np.vstack([header, row])


def collect_items(root: Path, product: str, max_n: int, seed: int) -> List[Tuple[Path, int, str]]:
    test_root = root / product / "test"
    items: List[Tuple[Path, int, str]] = []
    for d in sorted(p for p in test_root.iterdir() if p.is_dir()):
        gt_bad = 0 if d.name == "good" else 1
        for img in sorted(d.glob("*.png")):
            items.append((img, gt_bad, d.name))
    if max_n > 0 and len(items) > max_n:
        rng = random.Random(seed)
        items = rng.sample(items, max_n)
    return items


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    thr_cfg = load_thresholds(args.threshold_config)

    if args.decision_mode == "fused":
        cnn_ckpt = torch.load(args.cnn_weights, map_location=device)
        cnn_imgsz = int(cnn_ckpt.get("imgsz", 224))
        cnn = build_cnn(cnn_ckpt, device)
    else:
        cnn_ckpt = None
        cnn_imgsz = 224
        cnn = None

    args.out.mkdir(parents=True, exist_ok=True)

    global_cm = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    per_product = {}
    rows = []

    for product in args.products:
        if product not in thr_cfg.get("products", {}):
            print(f"[WARN] missing threshold config for {product}, skipping")
            continue

        pcfg = thr_cfg["products"][product]
        yolo_thr = float(pcfg["selected_threshold"])
        yolo_weights = pcfg["weights"]
        yolo = YOLO(yolo_weights)

        # Per-product fusion params override CLI defaults
        fusion_p = pcfg.get("fusion_params", {})
        p_alpha         = fusion_p.get("alpha",            args.alpha)
        p_fused_thr     = fusion_p.get("fused_threshold",  args.fused_threshold)
        p_min_bad_score = fusion_p.get("min_bad_score",    args.min_bad_score)
        p_heatmap_thr   = fusion_p.get("heatmap_threshold", args.heatmap_threshold)

        cm = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        items = collect_items(args.mvtec_root, product, args.max_images_per_product, args.seed)

        out_good = args.out / product / "pred_good"
        out_bad = args.out / product / "pred_bad"
        out_good.mkdir(parents=True, exist_ok=True)
        out_bad.mkdir(parents=True, exist_ok=True)

        for idx, (img_path, gt_bad, src_label) in enumerate(items):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            res = yolo.predict(source=str(img_path), conf=0.001, imgsz=args.imgsz, device=str(device), verbose=False)[0]
            if res.boxes is not None and len(res.boxes) > 0:
                yolo_score = float(np.max(res.boxes.conf.detach().cpu().numpy()))
            else:
                yolo_score = 0.0

            if yolo_thr <= 0:
                yolo_prob = yolo_score
            else:
                yolo_prob = float(np.clip(0.5 + 0.5 * (yolo_score - yolo_thr) / max(yolo_thr, 1e-6), 0.0, 1.0))

            c_prob = cnn_prob_bad(img, cnn, cnn_imgsz, device) if cnn is not None else 0.0
            fused = p_alpha * yolo_prob + (1.0 - p_alpha) * c_prob
            pred_bad = int(fused >= p_fused_thr)

            # Business rule: low-confidence detections should not be treated as BAD.
            if yolo_score < p_min_bad_score:
                pred_bad = 0

            add_cm(cm, gt_bad, pred_bad)
            add_cm(global_cm, gt_bad, pred_bad)

            if res.boxes is not None and len(res.boxes) > 0:
                boxes_xyxy = res.boxes.xyxy.detach().cpu().numpy()
                confs = res.boxes.conf.detach().cpu().numpy()
            else:
                boxes_xyxy = np.zeros((0, 4), dtype=np.float32)
                confs = np.zeros((0,), dtype=np.float32)

            # Hide low-confidence detections in visuals and pred heatmap
            if yolo_score < p_min_bad_score:
                vis = img.copy()
                boxes_for_heat = np.zeros((0, 4), dtype=np.float32)
                confs_for_heat = np.zeros((0,), dtype=np.float32)
            else:
                vis = res.plot()
                boxes_for_heat = boxes_xyxy
                confs_for_heat = confs

            if gt_bad == 1:
                mask_path = args.mvtec_root / product / "ground_truth" / src_label / f"{img_path.stem}_mask.png"
            else:
                mask_path = Path("__missing__")
            gt_mask = make_gt_mask(mask_path, (h, w))
            gt_heat = gt_mask.astype(np.float32)
            if np.max(gt_heat) > 0:
                gt_heat = gt_heat / 255.0

            pred_heat = build_pred_heatmap((h, w), boxes_for_heat, confs_for_heat)
            pred_heat_max = float(np.max(pred_heat))
            heat_bad = int(pred_heat_max >= p_heatmap_thr)
            if args.decision_mode == "heatmap_only":
                pred_bad = heat_bad
            elif args.decision_mode == "heatmap_priority":
                pred_bad = 1 if heat_bad else pred_bad
            gt_overlay = overlay_heatmap(img, gt_heat)
            pred_overlay = overlay_heatmap(img, pred_heat)

            txt = (
                f"GT={'BAD' if gt_bad else 'GOOD'} PRED={'BAD' if pred_bad else 'GOOD'} "
                f"y={yolo_score:.3f} yp={yolo_prob:.3f} c={c_prob:.3f} f={fused:.3f}"
            )
            panel = make_panel(img, vis, gt_overlay, pred_overlay, txt)
            dst = (out_bad if pred_bad else out_good) / f"{idx:04d}_{src_label}_{img_path.name}"
            cv2.imwrite(str(dst), panel)

            rows.append(
                {
                    "product": product,
                    "image": str(img_path),
                    "source_label": src_label,
                    "gt_bad": gt_bad,
                    "pred_bad": pred_bad,
                    "yolo_score": yolo_score,
                    "yolo_prob": yolo_prob,
                    "cnn_prob": c_prob,
                    "fused_score": fused,
                    "num_boxes": int(len(boxes_xyxy)),
                    "pred_heat_max": pred_heat_max,
                    "heat_bad": heat_bad,
                }
            )

        m = cm_metrics(cm)
        per_product[product] = {
            "confusion": cm,
            "metrics": m,
            "weights": yolo_weights,
            "yolo_threshold": yolo_thr,
            "fusion_params_used": {"alpha": p_alpha, "fused_threshold": p_fused_thr, "min_bad_score": p_min_bad_score, "heatmap_threshold": p_heatmap_thr},
        }
        print(
            f"[{product}] TP={cm['tp']} TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} "
            f"Acc={m['accuracy']:.4f} F1={m['f1_bad']:.4f} Rec={m['recall_bad']:.4f}"
        )

    g = cm_metrics(global_cm)
    report = {
        "settings": {
            "products": args.products,
            "alpha": args.alpha,
            "fused_threshold": args.fused_threshold,
            "device": str(device),
            "imgsz": args.imgsz,
            "cnn_weights": str(args.cnn_weights),
            "threshold_config": str(args.threshold_config),
        },
        "global": {"confusion": global_cm, "metrics": g},
        "per_product": per_product,
    }

    (args.out / "fusion_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    import csv

    with (args.out / "fusion_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "product",
                "image",
                "source_label",
                "gt_bad",
                "pred_bad",
                "yolo_score",
                "yolo_prob",
                "cnn_prob",
                "fused_score",
                "num_boxes",
                "pred_heat_max",
                "heat_bad",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"[global] TP={global_cm['tp']} TN={global_cm['tn']} FP={global_cm['fp']} FN={global_cm['fn']} "
        f"Acc={g['accuracy']:.4f} F1={g['f1_bad']:.4f} Rec={g['recall_bad']:.4f}"
    )
    print(f"Saved: {args.out / 'fusion_metrics.json'}")
    print(f"Saved: {args.out / 'fusion_predictions.csv'}")


if __name__ == "__main__":
    main()
