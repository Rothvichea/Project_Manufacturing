import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import yaml


IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")


@dataclass
class Item:
    path: Path
    label: int  # 0 good, 1 bad
    product: str
    defect_type: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build MVTec binary classification dataset (paper-style)")
    p.add_argument("--mvtec-root", type=Path, default=Path("data/mvtec"))
    p.add_argument(
        "--categories",
        nargs="+",
        default=["bottle", "screw", "transistor", "grid"],
        help="Products to include",
    )
    p.add_argument("--out", type=Path, default=Path("data/mvtec_cls_merged"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.6)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument(
        "--target-good-ratio",
        type=float,
        default=0.5,
        help="Target proportion of good images after balancing (paper used 0.5)",
    )
    p.add_argument(
        "--include-train-good",
        action="store_true",
        help="Also include <category>/train/good as negatives",
    )
    p.add_argument("--resize", type=int, default=224, help="Resize output images to NxN for paper style")
    p.add_argument("--clean", action="store_true")
    return p.parse_args()


def collect_items(root: Path, categories: Sequence[str], include_train_good: bool) -> List[Item]:
    items: List[Item] = []
    for cat in categories:
        test_root = root / cat / "test"
        if not test_root.exists():
            continue

        for defect_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
            label = 0 if defect_dir.name == "good" else 1
            for pattern in IMG_EXTS:
                for img in sorted(defect_dir.glob(pattern)):
                    items.append(Item(path=img, label=label, product=cat, defect_type=defect_dir.name))

        if include_train_good:
            train_good = root / cat / "train" / "good"
            if train_good.exists():
                for pattern in IMG_EXTS:
                    for img in sorted(train_good.glob(pattern)):
                        items.append(Item(path=img, label=0, product=cat, defect_type="train_good"))

    return items


def rebalance(items: Sequence[Item], target_good_ratio: float, seed: int) -> List[Item]:
    if not (0.0 < target_good_ratio < 1.0):
        raise ValueError("target-good-ratio must be in (0,1)")

    goods = [x for x in items if x.label == 0]
    bads = [x for x in items if x.label == 1]
    if not goods or not bads:
        return list(items)

    # good = r/(1-r) * bad
    max_goods = int(round((target_good_ratio / (1.0 - target_good_ratio)) * len(bads)))
    if max_goods >= len(goods):
        return list(items)

    rng = random.Random(seed)
    goods_sampled = rng.sample(goods, max_goods)
    return bads + goods_sampled


def split_group(group: Sequence[Item], train_ratio: float, val_ratio: float) -> Dict[str, List[Item]]:
    n = len(group)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": list(group[:n_train]),
        "val": list(group[n_train:n_train + n_val]),
        "test": list(group[n_train + n_val:]),
    }


def split_items(items: Sequence[Item], train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> Dict[str, List[Item]]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1")

    rng = random.Random(seed)
    goods = [x for x in items if x.label == 0]
    bads = [x for x in items if x.label == 1]
    rng.shuffle(goods)
    rng.shuffle(bads)

    g = split_group(goods, train_ratio, val_ratio)
    b = split_group(bads, train_ratio, val_ratio)

    out = {k: g[k] + b[k] for k in ("train", "val", "test")}
    for k in out:
        rng.shuffle(out[k])
    return out


def prepare_dirs(out: Path) -> None:
    for split in ("train", "val", "test"):
        for cls in ("good", "bad"):
            (out / split / cls).mkdir(parents=True, exist_ok=True)


def write_items(splits: Dict[str, List[Item]], out: Path, resize: int) -> Dict[str, Dict[str, int]]:
    counts = {s: {"good": 0, "bad": 0} for s in splits}
    idx = 0
    manifest = []

    for split, arr in splits.items():
        for it in arr:
            cls = "bad" if it.label == 1 else "good"
            dst = out / split / cls / f"img_{idx:06d}.png"
            idx += 1

            img = cv2.imread(str(it.path))
            if img is None:
                continue
            if resize > 0:
                img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dst), img)

            counts[split][cls] += 1
            manifest.append(
                {
                    "split": split,
                    "class": cls,
                    "product": it.product,
                    "defect_type": it.defect_type,
                    "src": str(it.path),
                    "dst": str(dst),
                }
            )

    with (out / "manifest.json").open("w", encoding="utf-8") as f:
        import json

        json.dump(manifest, f, indent=2)

    return counts


def write_yaml(out: Path, categories: Sequence[str], counts: Dict[str, Dict[str, int]], args: argparse.Namespace) -> None:
    build_args = {}
    for k, v in vars(args).items():
        build_args[k] = str(v) if isinstance(v, Path) else v

    meta = {
        "path": str(out.resolve()),
        "task": "binary_classification",
        "classes": {"0": "good", "1": "bad"},
        "splits": {
            "train": "train",
            "val": "val",
            "test": "test",
        },
        "categories": list(categories),
        "counts": counts,
        "build_args": build_args,
    }
    with (out / "dataset.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)


def main() -> None:
    args = parse_args()

    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    prepare_dirs(args.out)

    items = collect_items(args.mvtec_root, args.categories, args.include_train_good)
    if not items:
        raise RuntimeError("No items found; check mvtec root/categories")

    before_good = sum(1 for x in items if x.label == 0)
    before_bad = sum(1 for x in items if x.label == 1)

    items = rebalance(items, args.target_good_ratio, args.seed)

    after_good = sum(1 for x in items if x.label == 0)
    after_bad = sum(1 for x in items if x.label == 1)
    print(f"Balance: good {before_good}->{after_good}, bad {before_bad}->{after_bad}")

    splits = split_items(items, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    counts = write_items(splits, args.out, args.resize)
    write_yaml(args.out, args.categories, counts, args)

    print("\nClassification dataset created")
    for s in ("train", "val", "test"):
        n = counts[s]["good"] + counts[s]["bad"]
        print(f"{s:>5}: total={n:4d} good={counts[s]['good']:4d} bad={counts[s]['bad']:4d}")
    print(f"Output: {args.out}")
    print(f"Meta:   {args.out / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
