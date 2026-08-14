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
CONF = 0.3        # recall-first threshold (PLAN: prefer FP over FN)
CONF_LARGE = 0.1  # large objects (volumes/big holds) get a lower bar:
LARGE_AREA = 0.015  # bbox >= 1.5% of image area is "large" — volumes are physically
                    # big, and on dark walls they fire at ~0.15-0.2 conf (seen on
                    # 66ced028 blue volumes: 0.17 conf, 2.6% area, correctly missed
                    # by CONF but real). A per-size threshold beats a global one.


def preprocess(pil: Image.Image, res: int) -> np.ndarray:
    """letterbox to square, normalize (official RF-DETR transforms)."""
    from rfdetr.export.benchmark import infer_transforms

    tensor, _ = infer_transforms((res, res))(pil, None)
    return tensor[None].numpy()


def decode(out: list[np.ndarray], orig_w: int, orig_h: int, res: int):
    """RF-DETR ONNX outputs: dets = cxcywh normalized to the RES x RES STRETCHED
    input (infer_transforms does direct resize, NOT letterbox — verified by
    column-0 alignment). Map back with per-axis scale, no pad."""
    dets, labels, masks = out
    dets = dets[0]          # (100, 4) cxcywh, normalized to stretched input
    labels = labels[0]      # (100, 3) logits: [obj, cls0, cls1]
    masks = masks[0]        # (100, res/4, res/4) logits
    sx = orig_w / res       # stretched x scale (w/res)
    sy = orig_h / res       # stretched y scale (h/res)

    boxes, confs, cls_ids, mask_list = [], [], [], []
    for i in range(dets.shape[0]):
        conf = float(1 / (1 + np.exp(-labels[i][0])))  # obj logit -> sigmoid
        cls = 1 if labels[i][2] > labels[i][1] else 0  # argmax of class logits
        cx, cy, w, h = dets[i]
        thr = CONF_LARGE if (w * h) >= LARGE_AREA else CONF
        if conf < thr:
            continue
        # cxcywh normalized -> xyxy in ORIGINAL image pixels (per-axis scale)
        x1 = (cx - w / 2) * orig_w
        y1 = (cy - h / 2) * orig_h
        x2 = (cx + w / 2) * orig_w
        y2 = (cy + h / 2) * orig_h
        m = 1 / (1 + np.exp(-masks[i]))  # sigmoid
        m = cv2.resize(m, (res, res), interpolation=cv2.INTER_LINEAR)
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
        # smooth polygon outline (edge-preserving two-stage RDP), not raw mask pixels
        poly = mask_to_polygon(m)
        if poly is not None:
            pts = np.array(poly[0]).reshape(-1, 2).astype(np.int32)
            cv2.polylines(img, [pts], True, color, 2)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
        cv2.putText(img, f"{conf:.2f}", (int(x1), max(12, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


def session_resolution(sess) -> int:
    """Model input resolution from ONNX shape; fallback 384 on dynamic axes."""
    shape = sess.get_inputs()[0].shape
    if len(shape) == 4 and isinstance(shape[2], int):
        return shape[2]
    return RESOLUTION


MIN_AREA = 40      # px in original image
EPSILON = 1.5      # polygon simplification (px)


def mask_to_polygon(mask: np.ndarray, corner_window: int = 9,
                    corner_angle_deg: float = 30.0):
    """Mask -> edge-preserving polygon (pre-annotation friendly), two-stage:
    coarse RDP to drop raster steps, then corner-aware per-segment RDP so
    angles survive. Filter near-straight polygons (wall cracks/texture FP).
    """
    m = (mask > 0.5).astype(np.float32)
    m = cv2.GaussianBlur(m, (3, 3), 0)
    m = (m > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = max(contours, key=cv2.contourArea, default=None)
    if best is None or cv2.contourArea(best) < MIN_AREA:
        return None
    perim = cv2.arcLength(best, True)
    # stage 1: remove raster steps (eps ~1.5% of perimeter, min 2px)
    coarse = cv2.approxPolyDP(best, max(2.0, 0.015 * perim), True).reshape(-1, 2).astype(np.float32)
    pts = coarse
    n = len(pts)
    if n < 5:
        return [float(c) for xy in pts for c in xy], float(cv2.contourArea(best))

    # stage 2: corner detection on the smooth contour
    corners = set()
    for i in range(n):
        p0 = pts[(i - corner_window) % n]
        p1 = pts[(i + corner_window) % n]
        v1 = pts[i] - p0
        v2 = pts[i] - p1
        L1, L2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if L1 < 1e-6 or L2 < 1e-6:
            continue
        cos = float(np.clip(np.dot(v1, v2) / (L1 * L2), -1, 1))
        if np.degrees(np.arccos(cos)) < corner_angle_deg:
            corners.add(i)

    out: list[list[float]] = []
    if corners:
        order = sorted(corners)
        for a, b in zip(order, order[1:] + order[:1]):
            if b > a:
                seg = pts[a:b + 1]
            else:
                seg = np.vstack([pts[a:], pts[:b + 1]])
            seg = seg.reshape(-1, 1, 2)
            if len(seg) > 2:
                poly = cv2.approxPolyDP(seg, EPSILON, False).reshape(-1, 2)
                out.extend(poly.tolist())
    else:
        out = pts.tolist()

    if len(out) < 3:
        return None
    poly = np.array(out, np.float32)

    # Author-inspired filter (xiaoxiae std/utils.py): holds are never near-straight
    # lines — a near-linear polygon is a wall crack/edge/texture false positive.
    if len(poly) >= 4:
        xs, ys = poly[:, 0], poly[:, 1]
        slope, intercept = np.polyfit(xs, ys, 1)
        err = float(np.mean((slope * xs + intercept - ys) ** 2))
        if err < 5.0:
            return None

    return [float(c) for xy in poly for c in xy], float(cv2.contourArea(best))


def main(photos: Path, model_path: Path, out_dir: Path, tag: str) -> None:
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    res = session_resolution(sess)
    input_name = sess.get_inputs()[0].name
    out_dir = out_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"model={model_path.name} resolution={res} -> {out_dir}")

    for f in sorted(photos.glob("*.jpg")):
        pil = Image.open(f).convert("RGB")
        w, h = pil.size
        pixel = preprocess(pil, res)
        out = sess.run(None, {input_name: pixel})
        boxes, confs, cls_ids, masks = decode(out, w, h, res)
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
    ap.add_argument("--tag", type=str, default=None,
                    help="output subdir (default: model's parent dir name, e.g. onnx_640_aug)")
    args = ap.parse_args()
    tag = args.tag or args.model.parent.name
    main(args.photos, args.model, args.out, tag)
