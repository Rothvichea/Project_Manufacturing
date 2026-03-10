import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import yaml


@dataclass
class Sample:
    image_path: Path
    boxes: List[Tuple[float, float, float, float]]
    source: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a binary YOLO defect dataset from MVTec classes"
    )
    p.add_argument(
        "--mvtec-root",
        type=Path,
        default=Path("data/mvtec"),
        help="Path to MVTec root",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/mvtec_yolo_defect"),
        help="Output YOLO dataset directory",
    )
    p.add_argument(
        "--categories",
        nargs="+",
        default=["bottle", "screw", "transistor", "grid"],
        help="MVTec categories to include",
    )
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--min-area",
        type=int,
        default=8,
        help="Ignore tiny mask components below this pixel area",
    )
    p.add_argument(
        "--use-train-good",
        action="store_true",
        help="Include <category>/train/good as clean negatives",
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help="Remove output directory before generation",
    )
    p.add_argument(
        "--target-good-ratio",
        type=float,
        default=None,
        help=(
            "Optional target ratio of good images in final dataset (0.0-0.95). "
            "If current ratio is higher, good samples are downsampled."
        ),
    )
    return p.parse_args()


def ensure_layout(out_dir: Path) -> None:
    for split in ("train", "val", "test"):
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)


def mask_to_boxes(mask: cv2.UMat, min_area: int) -> List[Tuple[float, float, float, float]]:
    h, w = mask.shape[:2]
    _, bin_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    n, _, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)

    boxes: List[Tuple[float, float, float, float]] = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area or bw < 1 or bh < 1:
            continue
        cx = (x + bw / 2.0) / w
        cy = (y + bh / 2.0) / h
        nw = bw / w
        nh = bh / h
        boxes.append((cx, cy, nw, nh))
    return boxes


def collect_samples(
    mvtec_root: Path,
    categories: Sequence[str],
    min_area: int,
    include_train_good: bool,
) -> List[Sample]:
    samples: List[Sample] = []

    for cat in categories:
        base = mvtec_root / cat
        test_dir = base / "test"
        gt_dir = base / "ground_truth"

        if not test_dir.exists() or not gt_dir.exists():
            raise FileNotFoundError(f"Missing expected folders for category '{cat}'")

        for defect_dir in sorted(p for p in test_dir.iterdir() if p.is_dir() and p.name != "good"):
            for img_path in sorted(defect_dir.glob("*.png")):
                mask_path = gt_dir / defect_dir.name / f"{img_path.stem}_mask.png"
                if not mask_path.exists():
                    raise FileNotFoundError(f"Missing mask for {img_path}: {mask_path}")

                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise RuntimeError(f"Could not read mask: {mask_path}")

                boxes = mask_to_boxes(mask, min_area=min_area)
                if not boxes:
                    continue
                samples.append(Sample(image_path=img_path, boxes=boxes, source=f"{cat}/{defect_dir.name}"))

        for good_img in sorted((test_dir / "good").glob("*.png")):
            samples.append(Sample(image_path=good_img, boxes=[], source=f"{cat}/test/good"))

        if include_train_good:
            for good_img in sorted((base / "train" / "good").glob("*.png")):
                samples.append(Sample(image_path=good_img, boxes=[], source=f"{cat}/train/good"))

    return samples


def split_samples(
    samples: Sequence[Sample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    rng = random.Random(seed)
    defects = [s for s in samples if s.boxes]
    goods = [s for s in samples if not s.boxes]
    rng.shuffle(defects)
    rng.shuffle(goods)

    def assign(group: Sequence[Sample]) -> dict:
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        return {
            "train": list(group[:n_train]),
            "val": list(group[n_train:n_train + n_val]),
            "test": list(group[n_train + n_val:n_train + n_val + n_test]),
        }

    defect_splits = assign(defects)
    good_splits = assign(goods)

    splits = {
        k: defect_splits[k] + good_splits[k]
        for k in ("train", "val", "test")
    }
    for k in splits:
        rng.shuffle(splits[k])
    return splits


def rebalance_goods(samples: Sequence[Sample], target_good_ratio: float, seed: int) -> List[Sample]:
    if not (0.0 <= target_good_ratio < 1.0):
        raise ValueError("--target-good-ratio must be in [0.0, 1.0)")

    defects = [s for s in samples if s.boxes]
    goods = [s for s in samples if not s.boxes]
    if not defects or not goods:
        return list(samples)

    # target_good_ratio = G / (G + D)  =>  G = r/(1-r) * D
    max_goods = int(round((target_good_ratio / (1.0 - target_good_ratio)) * len(defects)))
    if max_goods >= len(goods):
        return list(samples)

    rng = random.Random(seed)
    sampled_goods = rng.sample(goods, k=max_goods)
    return defects + sampled_goods


def write_split(split_name: str, split_samples: Iterable[Sample], out_dir: Path, start_idx: int) -> int:
    idx = start_idx
    for sample in split_samples:
        stem = f"img_{idx:06d}"
        idx += 1

        dst_img = out_dir / split_name / "images" / f"{stem}.png"
        dst_lbl = out_dir / split_name / "labels" / f"{stem}.txt"

        shutil.copy2(sample.image_path, dst_img)

        with dst_lbl.open("w", encoding="utf-8") as f:
            for cx, cy, bw, bh in sample.boxes:
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    return idx


def write_dataset_yaml(out_dir: Path) -> None:
    data = {
        "path": str(out_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 1,
        "names": {0: "defect"},
    }
    with (out_dir / "dataset.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def summarize(splits: dict) -> None:
    print("\nDataset summary")
    print("=" * 44)
    for split_name in ("train", "val", "test"):
        items = splits[split_name]
        defects = sum(1 for s in items if s.boxes)
        goods = len(items) - defects
        boxes = sum(len(s.boxes) for s in items)
        print(f"{split_name:<5} images={len(items):>4} defects={defects:>4} good={goods:>4} boxes={boxes:>4}")


def main() -> None:
    args = parse_args()

    if args.clean and args.out.exists():
        shutil.rmtree(args.out)

    ensure_layout(args.out)

    samples = collect_samples(
        mvtec_root=args.mvtec_root,
        categories=args.categories,
        min_area=args.min_area,
        include_train_good=args.use_train_good,
    )
    if not samples:
        raise RuntimeError("No samples found. Check categories and mvtec path.")

    if args.target_good_ratio is not None:
        before_n = len(samples)
        before_g = sum(1 for s in samples if not s.boxes)
        before_d = before_n - before_g
        samples = rebalance_goods(samples, args.target_good_ratio, args.seed)
        after_n = len(samples)
        after_g = sum(1 for s in samples if not s.boxes)
        after_d = after_n - after_g
        print(
            f"Rebalanced goods: "
            f"before total={before_n} (defect={before_d}, good={before_g}) -> "
            f"after total={after_n} (defect={after_d}, good={after_g})"
        )

    splits = split_samples(
        samples=samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    idx = 0
    for split_name in ("train", "val", "test"):
        idx = write_split(split_name, splits[split_name], args.out, idx)

    write_dataset_yaml(args.out)
    summarize(splits)
    print(f"\nYOLO dataset written to: {args.out}")
    print(f"Dataset config: {args.out / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
