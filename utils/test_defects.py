import cv2
import numpy as np
import random
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generators.defect_generator import DefectGenerator

# Pick a random base image
img_dir = "data/mvtec/bottle/test/broken_large"
images  = os.listdir(img_dir)
src     = cv2.imread(f"{img_dir}/{random.choice(images)}")
src     = cv2.resize(src, (640, 640))

gen     = DefectGenerator()
defects = ["scratch", "dent", "contamination_oil",
           "contamination_rust", "contamination_dust"]

rows = []
for dtype in defects:
    row_imgs = [src.copy()]
    for sev in [0.2, 0.5, 0.9]:
        result, ann = gen.generate(src.copy(), dtype, severity=sev)

        # Draw bounding box
        h, w = result.shape[:2]
        cx, cy, bw, bh = ann.bbox
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        cv2.rectangle(result, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(result, f"sev={sev:.1f}", (x1, max(y1-5,10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        row_imgs.append(result)

    # Label column
    label = np.zeros((640, 160, 3), dtype=np.uint8)
    cv2.putText(label, dtype, (5, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
    rows.append(np.hstack([label] + row_imgs))

grid = np.vstack(rows)
out  = "data/output_previews/defect_test.jpg"
cv2.imwrite(out, grid)
print(f"✅ Preview saved → {out}")
print(f"   Grid size: {grid.shape[1]}x{grid.shape[0]} px")
print(f"   Open with:  xdg-open {out}")
