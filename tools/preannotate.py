"""Generate pre-annotations (COCO polygons) from the ONNX model for D1 labeling.

Model-predicted masks -> simplified polygons -> COCO JSON, ready for human
correction in Label Studio. Faster than labeling from scratch.

Usage:
    python tools/preannotate.py --photos data/realpic --model output/onnx_v101/crux-hold-seg-v1.0.1-648-fp16.onnx --out data/preannotations
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # allow `from infer_real import ...`
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from infer_real import CONF, preprocess, decode, session_resolution, mask_to_polygon  # reuse decode (same dir)

# mask_to_polygon lives in infer_real.py (shared with visualization)


def main(photos: Path, model_path: Path, out_dir: Path, tag: str) -> None:
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inp_res = session_resolution(sess)
    input_name = sess.get_inputs()[0].name
    out_dir = out_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"model={model_path.name} resolution={inp_res} -> {out_dir}")

    coco = {"images": [], "categories": [{"id": 0, "name": "hold"}, {"id": 1, "name": "volume"}],
            "annotations": []}
    ann_id = 0
    for img_id, f in enumerate(sorted(photos.glob("*.jpg"))):
        pil = Image.open(f).convert("RGB")
        w, h = pil.size
        out = sess.run(None, {input_name: preprocess(pil, inp_res)})
        boxes, confs, cls_ids, masks = decode(out, w, h, inp_res)
        coco["images"].append({"id": img_id, "file_name": f.name, "width": w, "height": h})
        for (x1, y1, x2, y2), conf, cls, m in zip(boxes, confs, cls_ids, masks):
            poly_res = mask_to_polygon(m)
            if poly_res is None:
                continue
            poly, area = poly_res
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
    ap.add_argument("--model", type=Path, default=Path("output/onnx_v101/crux-hold-seg-v1.0.1-648-fp16.onnx"))
    ap.add_argument("--out", type=Path, default=Path("data/preannotations"))
    ap.add_argument("--tag", type=str, default=None,
                    help="output subdir (default: model's parent dir name, e.g. onnx_640_aug)")
    args = ap.parse_args()
    tag = args.tag or args.model.parent.name
    main(args.photos, args.model, args.out, tag)
