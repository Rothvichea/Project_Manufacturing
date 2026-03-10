import argparse
import shutil
from pathlib import Path
from typing import List, Tuple

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge multiple YOLO datasets (same class mapping)")
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input YOLO dataset roots (each containing dataset.yaml + train/val/test)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/mvtec_yolo_merged"),
        help="Output merged dataset directory",
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help="Delete output directory before merge",
    )
    return p.parse_args()


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_layout(out_root: Path) -> None:
    for split in ("train", "val", "test"):
        (out_root / split / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split / "labels").mkdir(parents=True, exist_ok=True)


def collect_pairs(ds_root: Path, split: str) -> List[Tuple[Path, Path]]:
    img_dir = ds_root / split / "images"
    lbl_dir = ds_root / split / "labels"
    if not img_dir.exists() or not lbl_dir.exists():
        return []

    pairs: List[Tuple[Path, Path]] = []
    for img_path in sorted(p for p in img_dir.iterdir() if p.is_file()):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        pairs.append((img_path, lbl_path))
    return pairs


def write_dataset_yaml(out_root: Path, names: dict) -> None:
    data = {
        "path": str(out_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(names),
        "names": names,
    }
    with (out_root / "dataset.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main() -> None:
    args = parse_args()
    inputs = [Path(p) for p in args.inputs]

    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    ensure_layout(args.out)

    yamls = []
    for ds in inputs:
        y = ds / "dataset.yaml"
        if not y.exists():
            raise FileNotFoundError(f"Missing dataset.yaml in {ds}")
        yamls.append(read_yaml(y))

    base_names = yamls[0].get("names")
    base_nc = yamls[0].get("nc")
    for i, y in enumerate(yamls[1:], start=1):
        if y.get("names") != base_names or y.get("nc") != base_nc:
            raise ValueError(
                f"Class mapping mismatch between input[0] and input[{i}]. "
                "All inputs must share identical names/nc."
            )

    idx = 0
    counts = {"train": 0, "val": 0, "test": 0}
    for split in ("train", "val", "test"):
        for ds in inputs:
            pairs = collect_pairs(ds, split)
            for img_src, lbl_src in pairs:
                stem = f"img_{idx:06d}"
                idx += 1
                img_dst = args.out / split / "images" / f"{stem}{img_src.suffix.lower()}"
                lbl_dst = args.out / split / "labels" / f"{stem}.txt"
                shutil.copy2(img_src, img_dst)
                shutil.copy2(lbl_src, lbl_dst)
                counts[split] += 1

    write_dataset_yaml(args.out, base_names)

    print("Merged dataset summary")
    print("=" * 40)
    print(f"train images: {counts['train']}")
    print(f"val   images: {counts['val']}")
    print(f"test  images: {counts['test']}")
    print(f"Output: {args.out}")
    print(f"Config: {args.out / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
