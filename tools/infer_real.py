"""Run the research ONNX model on real gym photos and visualize results.

D1 exploration: how well does the 12-image research checkpoint generalize to
user's real photos? Produces annotated previews + per-image detection stats.

Usage:
    python tools/infer_real.py --photos data/realpic --model output/onnx/rfdetr-seg-small.onnx --out data/realpic_out
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

RESOLUTION = 384
CONF = 0.3  # recall-first threshold (PLAN: prefer FP over FN)


def preprocess(pil: Image.Image) -> np.ndarray:
    """letterbox to square, normalize (official RF-DETR transforms)."""
    from rfdetr.export.benchmark import infer_transforms

    tensor, _ = infer_transforms((RESOLUTION, RESOLUTION))(pil, None)
    return tensor[None].numpy()


def decode(out: list[np.ndarray], orig_w: int, orig_h: int):
    """RF-DETR ONNX outputs: dets = cxcywh normalized to the 384x384 STRETCHED
    input (infer_transforms does direct resize, NOT letterbox — verified by
    column-0 alignment). Map back with per-axis scale, no pad."""
    dets, labels, masks = out
    dets = dets[0]          # (100, 4) cxcywh, normalized to stretched 384x384
    labels = labels[0]      # (100, 3) logits: [obj, cls0, cls1]
    masks = masks[0]        # (100, 96, 96) logits
    sx = orig_w / RESOLUTION   # stretched x scale (w/384)
    sy = orig_h / RESOLUTION   # stretched y scale (h/384)

    boxes, confs, cls_ids, mask_list = [], [], [], []
    for i in range(dets.shape[0]):
        conf = float(1 / (1 + np.exp(-labels[i][0])))  # obj logit -> sigmoid
        cls = 1 if labels[i][2] > labels[i][1] else 0  # argmax of class logits
        if conf < CONF:
            continue
        cx, cy, w, h = dets[i]
        # cxcywh normalized -> xyxy in ORIGINAL image pixels (per-axis scale)
        x1 = (cx - w / 2) * orig_w
        y1 = (cy - h / 2) * orig_h
        x2 = (cx + w / 2) * orig_w
        y2 = (cy + h / 2) * orig_h
        m = 1 / (1 + np.exp(-masks[i]))  # sigmoid
        m = cv2.resize(m, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_LINEAR)
        m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)  # stretch back
        boxes.append([x1, y1, x2, y2])
        confs.append(conf)
        cls_ids.append(cls)
        mask_list.append(m)
    return boxes, confs, cls_ids, mask_list


def draw(img: np.ndarray, boxes, confs, cls_ids, masks):
    for (x1, y1, x2, y2), conf, cls, m in zip(boxes, confs, cls_ids, masks):
        color = (0, 200, 255) if cls == 0 else (255, 80, 255)  # hold / volume
        overlay = img.copy()
        overlay[m > 0.5] = overlay[m > 0.5] * 0.5 + np.array(color) * 0.5
        img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(img, f"{conf:.2f}", (int(x1), max(12, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


def main(photos: Path, model_path: Path, out_dir: Path) -> None:
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in sorted(photos.glob("*.jpg")):
        pil = Image.open(f).convert("RGB")
        w, h = pil.size
        pixel = preprocess(pil)
        out = sess.run(None, {input_name: pixel})
        boxes, confs, cls_ids, masks = decode(out, w, h)
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        annotated = draw(img, boxes, confs, cls_ids, masks)
        dest = out_dir / f"{f.stem}_annotated.jpg"
        cv2.imwrite(str(dest), annotated)
        if confs:
            n_hold = sum(1 for c in cls_ids if c == 0)
            n_vol = sum(1 for c in cls_ids if c == 1)
            print(f"{f.name}: {n_hold} holds, {n_vol} volumes, max_conf={max(confs):.2f}")
        else:
            print(f"{f.name}: 0 detections")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", type=Path, default=Path("data/realpic"))
    ap.add_argument("--model", type=Path, default=Path("output/onnx/rfdetr-seg-small.onnx"))
    ap.add_argument("--out", type=Path, default=Path("data/realpic_out"))
    args = ap.parse_args()
    main(args.photos, args.model, args.out)
