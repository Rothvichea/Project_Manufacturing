import argparse
import json
from pathlib import Path
from typing import Dict

from ultralytics import YOLO


VALID_PRODUCTS = {"bottle", "screw", "transistor", "grid"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Product-aware YOLO routing with per-product thresholds")
    p.add_argument("--source", type=str, required=True, help="Image/video/camera source for YOLO")
    p.add_argument(
        "--product",
        type=str,
        required=True,
        choices=sorted(VALID_PRODUCTS),
        help="Known product type for routing",
    )
    p.add_argument(
        "--threshold-config",
        type=Path,
        default=Path("validation/eval/Validation_and_Evaluation/thresholds_per_product.json"),
        help="Output from tune_product_thresholds.py",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="0")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("validation/eval/Validation_and_Evaluation/product_routing"),
    )
    p.add_argument("--save", action="store_true", help="Save YOLO visual outputs")
    p.add_argument("--stream", action="store_true", help="Use streaming mode for video/camera")
    return p.parse_args()


def load_cfg(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Threshold config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.threshold_config)

    if args.product not in cfg.get("products", {}):
        raise KeyError(f"Product '{args.product}' missing in threshold config")

    pcfg = cfg["products"][args.product]
    weights = pcfg["weights"]
    threshold = float(pcfg["selected_threshold"])

    model = YOLO(weights)

    print(f"Routing product='{args.product}'")
    print(f"Using weights: {weights}")
    print(f"Decision threshold: {threshold:.3f}")

    args.out.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=args.source,
        conf=0.001,  # collect full score range; final GOOD/BAD decided by tuned threshold
        imgsz=args.imgsz,
        device=args.device,
        stream=args.stream,
        save=args.save,
        project=str(args.out.parent),
        name=args.out.name,
        verbose=False,
    )

    # For non-stream source, ultralytics may return list; for stream it is iterator
    if not args.stream and isinstance(results, list):
        iterable = results
    else:
        iterable = results

    idx = 0
    for r in iterable:
        if r.boxes is not None and len(r.boxes) > 0:
            max_conf = float(r.boxes.conf.detach().cpu().numpy().max())
        else:
            max_conf = 0.0
        pred_bad = int(max_conf >= threshold)
        pred_label = "BAD" if pred_bad else "GOOD"
        print(f"frame={idx:04d} score={max_conf:.4f} pred={pred_label}")
        idx += 1


if __name__ == "__main__":
    main()
