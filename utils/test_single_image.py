"""
Single-image defect detection test with bounding boxes, heatmap overlay and GOOD/BAD verdict.

Usage:
    python utils/test_single_image.py --image <path> --product <bottle|screw|transistor|grid>

Example:
    python utils/test_single_image.py --image data/mvtec/bottle/test/broken_large/000.png --product bottle
    python utils/test_single_image.py --image data/raw_products/bottle_000.png --product bottle
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

THRESHOLD_CONFIG = Path("validation/results/thresholds_per_product.json")

PRODUCT_COLORS = {
    "bottle":     (78,  114, 176),
    "screw":      (85,  168, 104),
    "transistor": (196,  78,  82),
    "grid":       (221, 132,  82),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test defect detection on a single image")
    p.add_argument("--image",   type=Path, required=True,  help="Path to input image")
    p.add_argument("--product", type=str,  required=True,
                   choices=["bottle", "screw", "transistor", "grid"],
                   help="Product type")
    p.add_argument("--out-dir", type=Path, default=Path("validation/results/test_outputs"),
                   help="Where to save the annotated result image")
    p.add_argument("--imgsz",   type=int,  default=640)
    p.add_argument("--device",  type=str,  default="0",
                   help="GPU index or 'cpu'")
    return p.parse_args()


def resolve_device(dev: str) -> str:
    import torch
    d = dev.strip().lower()
    if d == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return f"cuda:{d}" if d.isdigit() else ("cuda:0" if d in ("gpu", "cuda") else d)
    return "cpu"


def build_heatmap(shape, boxes_xyxy: np.ndarray, confs: np.ndarray) -> np.ndarray:
    """Build a Gaussian-blurred confidence heatmap from YOLO boxes (same as fusion pipeline)."""
    h, w = shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    if boxes_xyxy.size == 0:
        return heat
    for (x1, y1, x2, y2), conf in zip(boxes_xyxy, confs):
        ix1 = int(np.clip(np.floor(x1), 0, w - 1))
        iy1 = int(np.clip(np.floor(y1), 0, h - 1))
        ix2 = int(np.clip(np.ceil(x2),  0, w - 1))
        iy2 = int(np.clip(np.ceil(y2),  0, h - 1))
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        heat[iy1:iy2, ix1:ix2] += float(conf)
    if np.max(heat) > 0:
        heat /= (np.max(heat) + 1e-8)
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=9, sigmaY=9)
    if np.max(heat) > 0:
        heat /= (np.max(heat) + 1e-8)
    return heat


def overlay_heatmap(img: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a [0,1] heatmap onto img using a red-yellow colormap."""
    heat_u8 = (heat * 255).astype(np.uint8)
    colormap = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    # Only blend where heat is non-trivial (> 2% of max range)
    mask = (heat > 0.02).astype(np.float32)[:, :, None]
    blended = (img.astype(np.float32) * (1 - alpha * mask) +
               colormap.astype(np.float32) * alpha * mask).clip(0, 255).astype(np.uint8)
    return blended


def draw_result(img: np.ndarray, boxes_xyxy: np.ndarray, confs: np.ndarray,
                all_boxes: np.ndarray, all_confs: np.ndarray,
                verdict: str, yolo_score: float, yolo_thr: float,
                product: str, color: tuple) -> np.ndarray:
    h, w = img.shape[:2]

    # ── heatmap overlay (built from ALL boxes, not just threshold-filtered) ──
    heat = build_heatmap(img.shape, all_boxes, all_confs)
    heat_max = float(np.max(heat))
    vis = overlay_heatmap(img.copy(), heat)

    # ── draw bounding boxes ──────────────────────────────────────────────
    for (x1, y1, x2, y2), conf in zip(boxes_xyxy, confs):
        ix1, iy1 = int(x1), int(y1)
        ix2, iy2 = int(x2), int(y2)
        box_color = (0, 0, 220) if verdict == "BAD" else (0, 200, 0)
        cv2.rectangle(vis, (ix1, iy1), (ix2, iy2), box_color, 2)
        label = f"defect {conf:.2f}"
        lw, lh = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(vis, (ix1, iy1 - lh - 6), (ix1 + lw + 4, iy1), box_color, -1)
        cv2.putText(vis, label, (ix1 + 2, iy1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # ── heatmap legend (bottom-right corner) ────────────────────────────
    legend_w, legend_h = 120, 14
    lx, ly = w - legend_w - 8, h - legend_h - 22
    grad = np.linspace(0, 255, legend_w, dtype=np.uint8)[None, :]
    grad_color = cv2.applyColorMap(np.tile(grad, (legend_h, 1)), cv2.COLORMAP_JET)
    vis[ly:ly+legend_h, lx:lx+legend_w] = grad_color
    cv2.rectangle(vis, (lx, ly), (lx+legend_w, ly+legend_h), (200,200,200), 1)
    cv2.putText(vis, "Low", (lx, ly+legend_h+12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)
    cv2.putText(vis, "High", (lx+legend_w-28, ly+legend_h+12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)
    cv2.putText(vis, "Defect heat", (lx+28, ly+legend_h+12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)

    # ── verdict banner at top ────────────────────────────────────────────
    banner_h = 54
    banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
    if verdict == "BAD":
        banner[:] = (0, 0, 160)
        verdict_text = "DEFECT DETECTED — BAD"
    else:
        banner[:] = (0, 100, 0)
        verdict_text = "NO DEFECT — GOOD"

    cv2.putText(banner, verdict_text,
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    info = (f"product={product}  YOLO={yolo_score:.3f} (thr={yolo_thr:.3f})"
            f"  boxes={len(boxes_xyxy)}  heat={heat_max:.3f}")
    cv2.putText(banner, info,
                (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    return np.vstack([banner, vis])


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    # ── load config ──────────────────────────────────────────────────────
    if not THRESHOLD_CONFIG.exists():
        raise FileNotFoundError(
            f"Config not found: {THRESHOLD_CONFIG}\n"
            "Make sure you have validation/results/thresholds_per_product.json"
        )
    cfg      = json.loads(THRESHOLD_CONFIG.read_text())
    pcfg     = cfg["products"][args.product]
    yolo_thr = float(pcfg["selected_threshold"])
    weights  = pcfg["weights"]
    fp       = pcfg.get("fusion_params", {})
    min_bad  = fp.get("min_bad_score", yolo_thr)

    # ── load image ───────────────────────────────────────────────────────
    img = cv2.imread(str(args.image))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")

    # ── run YOLO inference ───────────────────────────────────────────────
    yolo = YOLO(weights)
    res  = yolo.predict(source=str(args.image), conf=0.001,
                        imgsz=args.imgsz, device=device, verbose=False)[0]

    if res.boxes is not None and len(res.boxes) > 0:
        boxes_xyxy = res.boxes.xyxy.detach().cpu().numpy()
        confs      = res.boxes.conf.detach().cpu().numpy()
        yolo_score = float(np.max(confs))
        # only keep boxes above min_bad threshold for display
        keep = confs >= min_bad
        boxes_xyxy_show = boxes_xyxy[keep]
        confs_show      = confs[keep]
    else:
        boxes_xyxy = np.zeros((0, 4), dtype=np.float32)
        confs      = np.zeros((0,),   dtype=np.float32)
        yolo_score = 0.0
        boxes_xyxy_show = boxes_xyxy
        confs_show      = confs

    # ── decision ─────────────────────────────────────────────────────────
    verdict = "BAD" if yolo_score >= min_bad else "GOOD"

    # ── draw & save ──────────────────────────────────────────────────────
    color = PRODUCT_COLORS[args.product]
    out_img = draw_result(img, boxes_xyxy_show, confs_show,
                          boxes_xyxy, confs,
                          verdict, yolo_score, yolo_thr, args.product, color)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem     = args.image.stem
    out_path = args.out_dir / f"{stem}_{args.product}_{verdict}.png"
    cv2.imwrite(str(out_path), out_img)

    # ── print summary ─────────────────────────────────────────────────────
    print()
    print(f"  Product  : {args.product}")
    print(f"  Image    : {args.image}")
    print(f"  YOLO thr : {yolo_thr:.3f}  (min_bad_score={min_bad:.3f})")
    print(f"  YOLO max : {yolo_score:.4f}")
    print(f"  Boxes    : {len(boxes_xyxy_show)} defect region(s) shown")
    print(f"  Verdict  : {'🔴 BAD — defect detected' if verdict=='BAD' else '🟢 GOOD — no defect'}")
    print()
    print(f"  Saved    : {out_path}")
    print()


if __name__ == "__main__":
    main()
