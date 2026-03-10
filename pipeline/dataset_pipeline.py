import cv2, os, random, shutil, yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from generators.defect_generator import DefectGenerator, DEFECT_CLASSES

class DatasetPipeline:

    DEFECT_TYPES = [
        "scratch", "dent",
        "contamination_oil", "contamination_rust", "contamination_dust"
    ]

    def __init__(self, config_path="configs/defect_config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.gen = DefectGenerator()
        self.out = Path("data/generated")
        for split in ["train", "val", "test"]:
            (self.out / split / "images").mkdir(parents=True, exist_ok=True)
            (self.out / split / "labels").mkdir(parents=True, exist_ok=True)

    def run(self, source_dir="data/raw_products", total=2000):
        sources = [str(p) for p in Path(source_dir).glob("*.png")]
        sources += [str(p) for p in Path(source_dir).glob("*.jpg")]
        if not sources:
            raise RuntimeError(f"No images in {source_dir}")
        print(f"📂 Found {len(sources)} source images")

        # Split counts
        n_train = int(total * 0.70)
        n_val   = int(total * 0.20)
        n_test  = total - n_train - n_val
        splits  = {"train": n_train, "val": n_val, "test": n_test}

        clean_ratio = self.cfg["generation"]["clean_ratio"]
        img_size    = tuple(self.cfg["generation"]["image_size"])
        img_id = 0
        stats  = {s: {d: 0 for d in list(DEFECT_CLASSES.keys()) + ["clean"]}
                  for s in splits}

        for split, n in splits.items():
            n_clean    = int(n * clean_ratio)
            n_defect   = n - n_clean
            print(f"\n⚙️  Generating {split}: {n_defect} defective + {n_clean} clean")

            # --- defective images ---
            for _ in tqdm(range(n_defect), desc=f"  {split} defective"):
                src = cv2.imread(random.choice(sources))
                if src is None: continue
                src = cv2.resize(src, img_size)

                n_def = random.randint(*self.cfg["generation"]["defects_per_image"])
                anns  = []
                for _ in range(n_def):
                    dtype = random.choice(self.DEFECT_TYPES)
                    try:
                        src, ann = self.gen.generate(src, dtype)
                        anns.append(ann)
                        stats[split][dtype] += 1
                    except Exception as e:
                        continue

                # Light physics augmentation
                src = self._augment(src)
                self._save(src, anns, split, f"img_{img_id:06d}")
                img_id += 1

            # --- clean images ---
            for _ in tqdm(range(n_clean), desc=f"  {split} clean   "):
                src = cv2.imread(random.choice(sources))
                if src is None: continue
                src = cv2.resize(src, img_size)
                src = self._augment(src, light=True)
                self._save(src, [], split, f"img_{img_id:06d}")
                img_id += 1
                stats[split]["clean"] += 1

        self._save_yaml()
        self._print_stats(stats, total)
        return stats

    def _augment(self, img, light=False):
        """Simple physics augmentation inline"""
        # Gaussian noise
        sigma = random.uniform(2, 8) if light else random.uniform(3, 15)
        noise = np.random.normal(0, sigma, img.shape)
        img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)

        # Random brightness
        gamma = random.uniform(0.85, 1.15)
        img = np.clip(img.astype(float) * gamma, 0, 255).astype(np.uint8)

        # Occasional motion blur
        if not light and random.random() < 0.3:
            k = random.choice([3, 5])
            kernel = np.zeros((k, k))
            kernel[k//2, :] = 1.0 / k
            img = cv2.filter2D(img, -1, kernel)

        return img

    def _save(self, img, anns, split, name):
        cv2.imwrite(str(self.out / split / "images" / f"{name}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        label_path = self.out / split / "labels" / f"{name}.txt"
        with open(label_path, "w") as f:
            for ann in anns:
                cid = DEFECT_CLASSES.get(ann.defect_type, 0)
                cx, cy, w, h = ann.bbox
                f.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    def _save_yaml(self):
        d = {
            "path": str(self.out.absolute()),
            "train": "train/images",
            "val":   "val/images",
            "test":  "test/images",
            "nc":    len(DEFECT_CLASSES),
            "names": {v: k for k, v in DEFECT_CLASSES.items()}
        }
        with open(self.out / "dataset.yaml", "w") as f:
            yaml.dump(d, f, default_flow_style=False)
        print(f"\n📄 Dataset YAML → {self.out}/dataset.yaml")

    def _print_stats(self, stats, total):
        print("\n" + "="*45)
        print(f"✅ DATASET COMPLETE — {total} images")
        print("="*45)
        for split, counts in stats.items():
            n = sum(counts.values())
            print(f"\n  {split.upper()} ({n} images)")
            for k, v in counts.items():
                if v > 0:
                    print(f"    {k:<25} {v:>4}  ({100*v/n:.0f}%)")
        print(f"\n📁 Output: data/generated/")
        print(f"📁 YOLO config: data/generated/dataset.yaml")
