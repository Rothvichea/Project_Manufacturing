import cv2
import numpy as np
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class DefectAnnotation:
    defect_type: str
    bbox: Tuple[float, float, float, float]
    severity: float
    mask: np.ndarray = field(repr=False)

DEFECT_CLASSES = {
    "scratch": 0,
    "dent": 1,
    "contamination_oil": 2,
    "contamination_rust": 3,
    "contamination_dust": 4,
}

class DefectGenerator:

    def generate(self, image, defect_type, severity=None):
        if severity is None:
            severity = random.uniform(0.2, 0.9)

        # Detect product mask FIRST — only place defects on product
        product_mask = self._get_product_mask(image)

        if defect_type == "scratch":
            return self._scratch(image, severity, product_mask)
        elif defect_type == "dent":
            return self._dent(image, severity, product_mask)
        elif defect_type.startswith("contamination"):
            kind = defect_type.split("_")[1] if "_" in defect_type else "oil"
            return self._contamination(image, kind, severity, product_mask)
        else:
            raise ValueError(f"Unknown: {defect_type}")

    def _get_product_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Detect the product (non-background) region.
        Works by finding pixels significantly different from the background corner color.
        """
        h, w = image.shape[:2]

        # Sample background color from corners
        corners = [
            image[0:20, 0:20],
            image[0:20, w-20:w],
            image[h-20:h, 0:20],
            image[h-20:h, w-20:w],
        ]
        bg_color = np.median(
            np.vstack([c.reshape(-1, 3) for c in corners]), axis=0
        )

        # Find pixels different from background
        diff = np.abs(image.astype(float) - bg_color).mean(axis=2)
        _, product_mask = cv2.threshold(
            diff.astype(np.uint8), 20, 255, cv2.THRESH_BINARY
        )

        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        product_mask = cv2.morphologyEx(product_mask, cv2.MORPH_CLOSE, kernel)
        product_mask = cv2.morphologyEx(product_mask, cv2.MORPH_OPEN,
                                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))

        # If mask is too small (detection failed), use center region
        if product_mask.sum() < (h * w * 0.05 * 255):
            product_mask = np.zeros((h, w), dtype=np.uint8)
            margin_x, margin_y = w // 5, h // 5
            product_mask[margin_y:h-margin_y, margin_x:w-margin_x] = 255

        return product_mask

    def _get_product_center(self, product_mask):
        """Get a random point guaranteed to be ON the product"""
        ys, xs = np.where(product_mask > 0)
        if len(ys) == 0:
            h, w = product_mask.shape
            return w//2, h//2
        idx = random.randint(0, len(ys)-1)
        return int(xs[idx]), int(ys[idx])

    def _get_product_bbox(self, product_mask):
        """Get bounding box of the product region"""
        ys, xs = np.where(product_mask > 0)
        if len(ys) == 0:
            h, w = product_mask.shape
            return 0, 0, w, h
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    # ── SCRATCH ──────────────────────────────────────────────────
    def _scratch(self, image, severity, product_mask):
        h, w = image.shape[:2]
        result = image.copy()

        length = int(60 + severity * 180)
        width  = max(2, int(severity * 7))
        angle  = random.uniform(0, 180)
        rad    = np.radians(angle)

        # Start scratch from a point ON the product
        px0, py0 = self._get_product_center(product_mask)

        x1 = int(np.clip(px0 - (length/2)*np.cos(rad), 0, w-1))
        y1 = int(np.clip(py0 - (length/2)*np.sin(rad), 0, h-1))
        x2 = int(np.clip(px0 + (length/2)*np.cos(rad), 0, w-1))
        y2 = int(np.clip(py0 + (length/2)*np.sin(rad), 0, h-1))

        color_shift = int(-40 - severity * 60)
        all_pts = []

        for offset in range(-width//2, width//2 + 1):
            ox = int(offset * np.sin(rad))
            oy = int(offset * -np.cos(rad))
            for px, py in self._line_pixels(x1+ox, y1+oy, x2+ox, y2+oy):
                if 0 <= px < w and 0 <= py < h:
                    # Only draw on product pixels
                    if product_mask[py, px] > 0:
                        result[py, px] = np.clip(
                            result[py, px].astype(int) + color_shift, 0, 255)
                        all_pts.append((px, py))

        # Specular highlight
        for px, py in self._line_pixels(
            int(np.clip(x1+(width//2+2)*np.sin(rad), 0, w-1)),
            int(np.clip(y1+(width//2+2)*-np.cos(rad), 0, h-1)),
            int(np.clip(x2+(width//2+2)*np.sin(rad), 0, w-1)),
            int(np.clip(y2+(width//2+2)*-np.cos(rad), 0, h-1))
        ):
            if 0 <= px < w and 0 <= py < h and product_mask[py,px] > 0:
                result[py, px] = np.clip(result[py, px].astype(int)+30, 0, 255)

        mask = np.zeros((h, w), dtype=np.uint8)
        if all_pts:
            xs_drawn = [p[0] for p in all_pts]
            ys_drawn = [p[1] for p in all_pts]
            for px, py in all_pts:
                mask[py, px] = 255
            pad = max(5, width*2)
            x_min = max(0, min(xs_drawn)-pad)
            x_max = min(w-1, max(xs_drawn)+pad)
            y_min = max(0, min(ys_drawn)-pad)
            y_max = min(h-1, max(ys_drawn)+pad)
            bbox = (
                (x_min+x_max)/2/w,
                (y_min+y_max)/2/h,
                (x_max-x_min)/w,
                (y_max-y_min)/h,
            )
        else:
            bbox = (px0/w, py0/h, length/w, 0.05)

        return result, DefectAnnotation("scratch", bbox, severity, mask)

    # ── DENT ─────────────────────────────────────────────────────
    def _dent(self, image, severity, product_mask):
        h, w = image.shape[:2]
        result = image.copy().astype(float)

        radius = int(35 + severity * 75)

        # Find product pixels far enough from edges for full dent
        ys, xs = np.where(product_mask > 0)
        valid = [
            (x, y) for x, y in zip(xs, ys)
            if radius+2 < x < w-radius-2 and radius+2 < y < h-radius-2
        ]

        if valid:
            cx, cy = random.choice(valid)
        else:
            cx, cy = w//2, h//2

        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X-cx)**2 + (Y-cy)**2).astype(float)
        region = dist < radius

        # Only shade on product
        region_on_product = region & (product_mask > 0)

        depth = np.where(region_on_product,
                         (1-(dist/radius)**2) * severity, 0.0)
        shadow = depth * 80

        for c in range(3):
            result[:,:,c] = np.clip(result[:,:,c] - shadow, 0, 255)

        highlight = (dist < radius*0.4) & (X < cx) & (Y < cy) & (product_mask > 0)
        result[highlight] = np.clip(result[highlight]+35, 0, 255)

        # Bbox directly from dent center+radius
        x_min = max(0, cx-radius)
        y_min = max(0, cy-radius)
        x_max = min(w-1, cx+radius)
        y_max = min(h-1, cy+radius)

        bbox = (
            (x_min+x_max)/2/w,
            (y_min+y_max)/2/h,
            (x_max-x_min)/w,
            (y_max-y_min)/h,
        )

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[region_on_product] = 255

        return result.astype(np.uint8), DefectAnnotation("dent", bbox, severity, mask)

    # ── CONTAMINATION ────────────────────────────────────────────
    def _contamination(self, image, kind, severity, product_mask):
        h, w = image.shape[:2]
        result = image.copy().astype(float)
        mask = np.zeros((h, w), dtype=np.uint8)

        # Place blob center on product
        cx, cy = self._get_product_center(product_mask)
        area = int(h * w * (0.015 + severity * 0.08))

        if kind in ("oil", "rust"):
            blob = self._blob_mask_at(h, w, area, cx, cy)
            # Restrict to product only
            blob = cv2.bitwise_and(blob, product_mask)

            if kind == "oil":
                mean_b = float(np.mean(result[blob>0])) if blob.any() else 128
                if mean_b < 80:
                    for c, add in enumerate([15, 20, 30]):
                        result[:,:,c] = np.where(blob>0,
                            np.clip(result[:,:,c]+add*severity,0,255),
                            result[:,:,c])
                else:
                    for c, tint in enumerate([0.72, 0.82, 0.95]):
                        result[:,:,c] = np.where(blob>0,
                            np.clip(result[:,:,c]*tint,0,255),
                            result[:,:,c])
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
                edge = cv2.dilate(blob,kernel) - blob
                result[:,:,2] = np.where(edge>0,
                    np.clip(result[:,:,2]+25,0,255), result[:,:,2])

            else:  # rust
                rust_bgr = np.array([25, 55, 160])
                alpha = 0.4 + severity * 0.45
                for c in range(3):
                    result[:,:,c] = np.where(blob>0,
                        np.clip(result[:,:,c]*(1-alpha)+rust_bgr[c]*alpha,0,255),
                        result[:,:,c])
            mask = blob

        elif kind == "dust":
            patch_r = int(35 + severity * 80)
            n_particles = int(50 + severity * 150)
            tmp = result.copy().astype(np.uint8)
            px_list, py_list = [], []

            for _ in range(n_particles):
                angle = random.uniform(0, 2*np.pi)
                d = random.uniform(0, patch_r)
                px = int(np.clip(cx + d*np.cos(angle), 0, w-1))
                py = int(np.clip(cy + d*np.sin(angle), 0, h-1))
                # Only draw on product
                if product_mask[py, px] > 0:
                    b = random.randint(185, 255)
                    r = random.randint(2, 5)
                    cv2.circle(tmp, (px,py), r, (b,b,b), -1)
                    cv2.circle(mask, (px,py), r, 255, -1)
                    px_list.append(px)
                    py_list.append(py)

            result = tmp.astype(float)

            if px_list:
                pad = 10
                x_min = max(0, min(px_list)-pad)
                x_max = min(w-1, max(px_list)+pad)
                y_min = max(0, min(py_list)-pad)
                y_max = min(h-1, max(py_list)+pad)
                bbox = (
                    (x_min+x_max)/2/w,
                    (y_min+y_max)/2/h,
                    (x_max-x_min)/w,
                    (y_max-y_min)/h,
                )
                return result.astype(np.uint8), DefectAnnotation(
                    "contamination_dust", bbox, severity, mask)

        bbox = self._mask_to_bbox(mask, w, h)
        return result.astype(np.uint8), DefectAnnotation(
            f"contamination_{kind}", bbox, severity, mask)

    # ── HELPERS ──────────────────────────────────────────────────
    def _line_pixels(self, x1, y1, x2, y2):
        pts = []
        dx, dy = abs(x2-x1), abs(y2-y1)
        sx = 1 if x1<x2 else -1
        sy = 1 if y1<y2 else -1
        err = dx-dy
        while True:
            pts.append((x1,y1))
            if x1==x2 and y1==y2: break
            e2 = 2*err
            if e2>-dy: err-=dy; x1+=sx
            if e2< dx: err+=dx; y1+=sy
        return pts

    def _blob_mask_at(self, h, w, target_area, cx, cy):
        """Blob centered at specific location"""
        mask = np.zeros((h,w), dtype=np.uint8)
        n = random.randint(4, 10)
        r_avg = max(8, int(np.sqrt(target_area/(np.pi*n))))
        for _ in range(n):
            r = random.randint(max(4, r_avg//2), r_avg*2)
            x = cx + random.randint(-r_avg, r_avg)
            y = cy + random.randint(-r_avg, r_avg)
            x = int(np.clip(x, 0, w-1))
            y = int(np.clip(y, 0, h-1))
            cv2.circle(mask, (x,y), r, 255, -1)
        return mask

    def _mask_to_bbox(self, mask, w, h):
        coords = np.where(mask>0)
        if len(coords[0])==0:
            return (0.5,0.5,0.1,0.1)
        y_min,y_max = int(coords[0].min()),int(coords[0].max())
        x_min,x_max = int(coords[1].min()),int(coords[1].max())
        return (
            float((x_min+x_max)/2/w),
            float((y_min+y_max)/2/h),
            float(np.clip((x_max-x_min)/w, 0.01, 1)),
            float(np.clip((y_max-y_min)/h, 0.01, 1)),
        )
