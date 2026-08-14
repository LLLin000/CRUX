"""Generate pre-annotations (COCO polygons) from the ONNX model for D1 labeling.

Model-predicted masks -> simplified polygons -> COCO JSON, ready for human
correction in Label Studio. Faster than labeling from scratch.

Usage:
    python tools/preannotate.py --photos data/realpic --model output/onnx/rfdetr-seg-small.onnx --out data/preannotations
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from infer_real import RESOLUTION, CONF, preprocess, decode  # reuse decode (same dir)

MIN_AREA = 40      # px in original image
EPSILON = 1.5      # polygon simplification (px)


def mask_to_polygon(mask: np.ndarray, epsilon: float = EPSILON):
    m = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = max(contours, key=cv2.contourArea, default=None)
    if best is None or cv2.contourArea(best) < MIN_AREA:
        return None
    poly = cv2.approxPolyDP(best, epsilon, True).reshape(-1, 2)
    if len(poly) < 3:
        return None
    return [float(c) for xy in poly for c in xy], float(cv2.contourArea(best))


def main(photos: Path, model_path: Path, out_dir: Path) -> None:
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = {"images": [], "categories": [{"id": 0, "name": "hold"}, {"id": 1, "name": "volume"}],
            "annotations": []}
    ann_id = 0
    for img_id, f in enumerate(sorted(photos.glob("*.jpg"))):
        pil = Image.open(f).convert("RGB")
        w, h = pil.size
        out = sess.run(None, {input_name: preprocess(pil)})
        boxes, confs, cls_ids, masks = decode(out, w, h)
        coco["images"].append({"id": img_id, "file_name": f.name, "width": w, "height": h})
        for (x1, y1, x2, y2), conf, cls, m in zip(boxes, confs, cls_ids, masks):
            res = mask_to_polygon(m)
            if res is None:
                continue
            poly, area = res
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": cls,
                "segmentation": [[float(c) for c in poly]], "area": round(float(area), 1),
                "bbox": [round(float(x1), 1), round(float(y1), 1),
                         round(float(x2 - x1), 1), round(float(y2 - y1), 1)],
                "iscrowd": 0,
                "preannotated": True, "confidence": round(float(conf), 2),
            })
            ann_id += 1
        print(f"{f.name}: {sum(1 for c in cls_ids if c == 0)} holds, "
              f"{sum(1 for c in cls_ids if c == 1)} volumes -> {ann_id} total anns")

    dest = out_dir / "preannotations.json"
    dest.write_text(json.dumps(coco), encoding="utf-8")
    print(f"wrote {dest} ({len(coco['annotations'])} annotations, {len(coco['images'])} images)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", type=Path, default=Path("data/realpic"))
    ap.add_argument("--model", type=Path, default=Path("output/onnx/rfdetr-seg-small.onnx"))
    ap.add_argument("--out", type=Path, default=Path("data/preannotations"))
    args = ap.parse_args()
    main(args.photos, args.model, args.out)
