# Manufacturing Defect Detection

A production-ready defect detection pipeline for industrial parts using **YOLOv8** with a **Gaussian confidence heatmap** overlay. Trained and evaluated on the [MVTec Anomaly Detection Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad).

---

## Results

| Product | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|
| Bottle | 0.984 | 0.984 | 0.984 | 0.976 |
| Screw | 0.944 | 0.973 | 0.916 | 0.919 |
| Transistor | 0.919 | 1.000 | 0.850 | 0.940 |
| Grid | **1.000** | **1.000** | **1.000** | **1.000** |
| **Global** | **0.961** | **0.985** | **0.939** | **0.950** |

---

## How It Works

```
Input Image
    │
    ▼
YOLOv8 (per-product fine-tuned)
    │  bounding boxes + confidence scores
    ▼
Gaussian Confidence Heatmap
    │  spatial defect probability map
    ▼
heatmap_priority decision rule
    │  if heatmap fires → BAD, else use YOLO score
    ▼
GOOD / BAD verdict
```

Each product has its own fine-tuned YOLOv8 model and tuned decision threshold optimised for F1 score with a minimum recall constraint of 75%.

---

## Project Structure

```
Project_Manufacturing/
├── configs/                        # Training & defect configs
├── data/
│   ├── mvtec_yolo_bottle/          # YOLO-format dataset (bottle)
│   ├── mvtec_yolo_screw/           # YOLO-format dataset (screw)
│   ├── mvtec_yolo_transistor/      # YOLO-format dataset (transistor)
│   └── mvtec_yolo_grid/            # YOLO-format dataset (grid)
├── generators/                     # Synthetic defect generator
├── pipeline/                       # Data preparation scripts
│   ├── mvtec_to_yolo.py            # Convert MVTec → YOLO format
│   ├── mvtec_to_cls.py             # Convert MVTec → classifier format
│   ├── merge_yolo_datasets.py      # Merge product datasets
│   └── dataset_pipeline.py        # End-to-end data pipeline
├── utils/
│   ├── test_single_image.py        # Run inference on one image
│   ├── eval_fusion_mvtec.py        # Evaluate on full MVTec test set
│   ├── tune_product_thresholds.py  # Optimise decision thresholds
│   ├── fuse_yolo_cnn_decision.py   # YOLO + CNN fusion logic
│   └── predict_product_routing.py  # Route image to correct model
├── runs/                           # Trained weights (best.pt per product)
├── validation/results/             # Metrics, threshold configs, plots
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/Rothvichea/Project_Manufacturing.git
cd Project_Manufacturing

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download the [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) and extract it to `data/mvtec/`.

---

## Usage

**Test a single image:**
```bash
python utils/test_single_image.py \
  --image data/mvtec/bottle/test/broken_large/000.png \
  --product bottle
```

**Evaluate all 4 products:**
```bash
python utils/eval_fusion_mvtec.py \
  --products bottle screw transistor grid \
  --threshold-config validation/results/thresholds_per_product.json \
  --decision-mode heatmap_priority
```

---

## Trained Weights

| Product | Path |
|---|---|
| Bottle | `runs/detect/validation/runs/mvtec_bottle_binary_v16/weights/best.pt` |
| Screw | `runs/detect/validation/runs/mvtec_screw_binary_v4/weights/best.pt` |
| Transistor | `runs/detect/validation/runs/mvtec_transistor_binary_v23/weights/best.pt` |
| Grid | `runs/detect/validation/runs/mvtec_grid_binary_v1/weights/best.pt` |

---

## Tech Stack

- [YOLOv8](https://github.com/ultralytics/ultralytics) — object detection backbone
- OpenCV — image processing & heatmap overlay
- PyTorch — model training & inference
- MVTec AD — industrial defect benchmark dataset
