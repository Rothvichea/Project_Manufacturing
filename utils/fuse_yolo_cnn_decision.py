import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fuse per-product YOLO + paper-style CNN verdict")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--product", type=str, required=True, choices=["bottle", "screw", "transistor", "grid"])
    p.add_argument(
        "--threshold-config",
        type=Path,
        default=Path("validation/eval/Validation_and_Evaluation/thresholds_per_product.json"),
    )
    p.add_argument(
        "--cnn-weights",
        type=Path,
        default=Path("validation/runs/vgg16_cls_merged_v1/best.pt"),
        help="Weights from train_vgg16_classifier.py",
    )
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--alpha", type=float, default=0.6, help="Fusion weight: alpha*YOLO + (1-alpha)*CNN")
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
        help="Pred heatmap max >= threshold is considered abnormal.",
    )
    p.add_argument("--out", type=Path, default=Path("validation/eval/Validation_and_Evaluation/fusion_single"))
    return p.parse_args()


def load_threshold_cfg(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def build_cnn_from_ckpt(ckpt: Dict, device: torch.device) -> nn.Module:
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


def run_cnn_prob_bad(img_bgr: np.ndarray, model: nn.Module, imgsz: int, device: torch.device) -> float:
    tf = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((imgsz, imgsz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    x = tf(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(x).squeeze(1)
        prob_bad = torch.sigmoid(logit).item()
    return float(prob_bad)


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


def main() -> None:
    args = parse_args()
    dev_arg = str(args.device).strip().lower()
    if dev_arg == "cpu":
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        if dev_arg.isdigit():
            device = torch.device(f"cuda:{dev_arg}")
        elif dev_arg.startswith("cuda"):
            device = torch.device(dev_arg)
        else:
            device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    cfg = load_threshold_cfg(args.threshold_config)
    prod_cfg = cfg["products"][args.product]
    yolo_weights = prod_cfg["weights"]
    yolo_thr = float(prod_cfg["selected_threshold"])

    # Per-product fusion params override CLI defaults
    fusion_p = prod_cfg.get("fusion_params", {})
    p_alpha        = fusion_p.get("alpha",            args.alpha)
    p_fused_thr    = fusion_p.get("fused_threshold",  args.fused_threshold)
    p_heatmap_thr  = fusion_p.get("heatmap_threshold", args.heatmap_threshold)

    yolo = YOLO(yolo_weights)

    cnn_ckpt = torch.load(args.cnn_weights, map_location=device)
    cnn_imgsz = int(cnn_ckpt.get("imgsz", 224))
    cnn = build_cnn_from_ckpt(cnn_ckpt, device)

    img = cv2.imread(str(args.image))
    if img is None:
        raise RuntimeError(f"Cannot read image: {args.image}")

    # YOLO score: max defect confidence
    res = yolo.predict(source=str(args.image), conf=0.001, imgsz=args.imgsz, device=str(device), verbose=False)[0]
    if res.boxes is not None and len(res.boxes) > 0:
        confs = res.boxes.conf.detach().cpu().numpy()
        boxes_xyxy = res.boxes.xyxy.detach().cpu().numpy()
        yolo_score = float(np.max(confs))
    else:
        confs = np.zeros((0,), dtype=np.float32)
        boxes_xyxy = np.zeros((0, 4), dtype=np.float32)
        yolo_score = 0.0

    # Normalize yolo score around tuned threshold to pseudo-probability around 0.5
    # score>=thr => yolo_prob>=0.5 approximately
    if yolo_thr <= 0:
        yolo_prob = yolo_score
    else:
        yolo_prob = float(np.clip(0.5 + 0.5 * (yolo_score - yolo_thr) / max(yolo_thr, 1e-6), 0.0, 1.0))

    cnn_prob = run_cnn_prob_bad(img, cnn, cnn_imgsz, device)
    pred_heat = build_pred_heatmap(img.shape[:2], boxes_xyxy, confs)
    pred_heat_max = float(np.max(pred_heat))
    heat_bad = int(pred_heat_max >= p_heatmap_thr)

    fused = p_alpha * yolo_prob + (1.0 - p_alpha) * cnn_prob
    pred_bad = int(fused >= p_fused_thr)
    if args.decision_mode == "heatmap_only":
        pred_bad = heat_bad
    elif args.decision_mode == "heatmap_priority":
        pred_bad = 1 if heat_bad else pred_bad

    vis = res.plot()
    txt = (
        f"product={args.product} YOLO={yolo_score:.3f}(thr={yolo_thr:.3f}) "
        f"YOLOp={yolo_prob:.3f} CNNp={cnn_prob:.3f} heat={pred_heat_max:.3f} fused={fused:.3f} "
        f"pred={'BAD' if pred_bad else 'GOOD'}"
    )
    cv2.putText(vis, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    args.out.mkdir(parents=True, exist_ok=True)
    out_img = args.out / f"{args.image.stem}_fusion.png"
    out_json = args.out / f"{args.image.stem}_fusion.json"
    cv2.imwrite(str(out_img), vis)
    out_json.write_text(
        json.dumps(
            {
                "product": args.product,
                "image": str(args.image),
                "yolo_weights": yolo_weights,
                "yolo_threshold": yolo_thr,
                "yolo_score": yolo_score,
                "yolo_prob": yolo_prob,
                "cnn_weights": str(args.cnn_weights),
                "cnn_prob": cnn_prob,
                "pred_heat_max": pred_heat_max,
                "heat_bad": bool(heat_bad),
                "decision_mode": args.decision_mode,
                "heatmap_threshold": p_heatmap_thr,
                "alpha": p_alpha,
                "fused_score": fused,
                "fused_threshold": p_fused_thr,
                "pred": "BAD" if pred_bad else "GOOD",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(txt)
    print(f"Saved: {out_img}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
