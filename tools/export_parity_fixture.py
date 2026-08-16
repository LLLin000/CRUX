"""Export Swift/Python parity fixtures (tools/export_parity_fixture.py).

Freezes the Python reference implementation so the Swift Core can be verified
layer by layer on the same inputs:

1. preprocess: synthetic pattern -> torchvision v2 reference tensor
   (ToImage -> /255 -> bilinear Resize -> ImageNet Normalize -> CHW).
2. decode: real photo -> FP16 ONNX -> dets/labels/masks outputs + Python
   decode reference (conf/class/bbox + bbox-local mask RLE via the same
   two-stage bilinear + clip + threshold pipeline).
3. lab: synthetic pattern + two synthetic full-bbox holds -> median Lab
   reference using the exact Swift sampling/median semantics.

Outputs under Tests/CRUXCoreTests/Fixtures/:
  parity_synth_rgba.bin          RGBA8 synthetic pattern (W*H*4 bytes)
  parity_preprocess_synth.bin    float32 CHW reference tensor (3,size,size)
  parity_model_outputs.bin       float32 [dets(400) + labels(300) + masks(100,162,162)]
  parity_decode_reference.json   per-hold conf/class/bbox + mask RLE
  parity_lab_reference.json      two synthetic holds + median Lab

Usage:
  python tools/export_parity_fixture.py [--selfcheck]
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "Tests" / "CRUXCoreTests" / "Fixtures"
MODEL = ROOT / "output" / "onnx_v101" / "crux-hold-seg-v1.0.1-648-fp16.onnx"
PHOTO = ROOT / "data" / "realpic" / "017cd2184be92752b40c38f51616b89a.jpg"
SYNTH_W, SYNTH_H = 200, 150
INPUT_SIZE = 648


# ---------------------------------------------------------------- synthetic

def synth_rgba() -> bytes:
    """Deterministic 200x150 pattern: gradient + two solid blocks."""
    out = bytearray(SYNTH_W * SYNTH_H * 4)
    for y in range(SYNTH_H):
        for x in range(SYNTH_W):
            r = (x * 255) // SYNTH_W
            g = (y * 255) // SYNTH_H
            b = 128 + ((x // 40) * 32) % 128
            if 20 <= x < 80 and 30 <= y < 90:
                r, g, b = 220, 30, 40
            if 120 <= x < 180 and 60 <= y < 130:
                r, g, b = 30, 120, 220
            i = (y * SYNTH_W + x) * 4
            out[i:i + 4] = bytes((r, g, b, 255))
    return bytes(out)


def reference_preprocess(rgba: bytes) -> np.ndarray:
    from PIL import Image
    from rfdetr.export.benchmark import infer_transforms
    img = Image.frombytes("RGBA", (SYNTH_W, SYNTH_H), rgba).convert("RGB")
    tensor, _ = infer_transforms((INPUT_SIZE, INPUT_SIZE))(img, None)
    # transform returns (3, H, W) CHW (no batch dim)
    return np.ascontiguousarray(tensor.numpy(), dtype=np.float32)


# ------------------------------------------------------------------ decode

def rle_encode(mask: np.ndarray) -> list[int]:
    """Row-major binary bitmap -> COCO counts (0-run first), int32 values."""
    flat = mask.ravel()
    counts: list[int] = []
    value = 0
    run = 0
    for bit in flat:
        if bit == value:
            run += 1
        else:
            counts.append(run)
            value = bit
            run = 1
    counts.append(run)
    return counts


def python_decode_reference() -> dict:
    """Run the reference pipeline on a real photo and freeze its output.

    Self-contained filtering loop (same semantics as `infer_real.decode`)
    so that queryIndex, bbox, and mask RLE come from the SAME dets row.
    Matching boxes back to rows afterwards is ambiguous when RF-DETR emits
    duplicate queries, which is why we do not reuse `decode`'s list order.
    """
    import onnxruntime as ort
    from PIL import Image
    sys.path.insert(0, str(ROOT / "tools"))
    from infer_real import preprocess  # noqa: E402

    img = Image.open(PHOTO).convert("RGB")
    x = preprocess(img, INPUT_SIZE)
    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    dets, labels, masks = sess.run(None, {"input": x})
    dets, labels, masks = dets[0], labels[0], masks[0]
    (FIXTURES / "parity_model_outputs.bin").write_bytes(
        dets.astype(np.float32).tobytes()
        + labels.astype(np.float32).tobytes()
        + masks.astype(np.float32).tobytes()
    )

    W, H = img.width, img.height
    holds = []
    for i in range(dets.shape[0]):
        conf = float(1 / (1 + np.exp(-np.clip(labels[i][0], -60, 60))))
        cx, cy, w, h = dets[i]
        thr = 0.10 if w * h >= 0.015 else 0.30
        if conf < thr:
            continue
        x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
        x2, y2 = (cx + w / 2) * W, (cy + h / 2) * H
        m = 1 / (1 + np.exp(-np.clip(masks[i], -60, 60)))
        m = cv2.resize(m, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
        bx1 = max(0, min(W, int(math.floor(x1))))
        by1 = max(0, min(H, int(math.floor(y1))))
        bx2 = max(0, min(W, int(math.ceil(x2))))
        by2 = max(0, min(H, int(math.ceil(y2))))
        clipped = np.zeros_like(m)
        if bx2 > bx1 and by2 > by1:
            clipped[by1:by2, bx1:bx2] = m[by1:by2, bx1:bx2]
        m = clipped
        nx1 = max(0.0, min(1.0, x1 / W))
        ny1 = max(0.0, min(1.0, y1 / H))
        nx2 = max(0.0, min(1.0, x2 / W))
        ny2 = max(0.0, min(1.0, y2 / H))
        local = (m[by1:by2, bx1:bx2] > 0.5).astype(np.uint8)
        holds.append({
            "queryIndex": int(i),
            "conf": round(conf, 6),
            "cls": int(1 if labels[i][2] > labels[i][1] else 0),
            "x1": round(float(nx1), 9), "y1": round(float(ny1), 9),
            "x2": round(float(nx2), 9), "y2": round(float(ny2), 9),
            "maskW": int(bx2 - bx1), "maskH": int(by2 - by1),
            "rle": rle_encode(local),
        })
    return {"imageWidth": W, "imageHeight": H, "holds": holds}


# --------------------------------------------------------------------- lab

def srgb_to_lab(r: int, g: int, b: int) -> tuple[float, float, float]:
    def lin(c: float) -> float:
        c01 = c / 255.0
        return math.pow((c01 + 0.055) / 1.055, 2.4) if c01 > 0.04045 else c01 / 12.92
    rl, gl, bl = lin(r), lin(g), lin(b)
    x = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl
    y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl
    z = 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl
    xn, yn, zn = 0.95047, 1.0, 1.08883
    d = 6.0 / 29.0

    def f(t: float) -> float:
        return math.cbrt(t) if t > d * d * d else t / (3 * d * d) + 4.0 / 29.0
    fx, fy, fz = f(x / xn), f(y / yn), f(z / zn)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def median_lab(rgba: bytes, hold: dict) -> dict:
    """Swift-identical sampling: bbox-local RLE -> normalized -> floor pixel."""
    w, h = SYNTH_W, SYNTH_H
    labs = []
    counts = hold["rle"]
    idx = 0
    value = 0
    for count in counts:
        for _ in range(count):
            if value == 1:
                local_x = idx % hold["maskW"]
                local_y = idx // hold["maskW"]
                nx = hold["bboxX"] + (local_x + 0.5) / hold["maskW"] * hold["bboxW"]
                ny = hold["bboxY"] + (local_y + 0.5) / hold["maskH"] * hold["bboxH"]
                px, py = int(nx * w), int(ny * h)
                i = (py * w + px) * 4
                labs.append(srgb_to_lab(rgba[i], rgba[i + 1], rgba[i + 2]))
            idx += 1
        value = 1 - value
    if not labs:
        return {"l": 50, "a": 0, "b": 0}
    med = []
    for channel in range(3):
        v = sorted(x[channel] for x in labs)
        mid = len(v) // 2
        med.append(v[mid] if len(v) % 2 == 1 else (v[mid - 1] + v[mid]) / 2)
    return {"l": round(med[0], 6), "a": round(med[1], 6), "b": round(med[2], 6)}


def lab_reference(rgba: bytes) -> dict:
    holds = [
        {"bboxX": 0.10, "bboxY": 0.20, "bboxW": 0.30, "bboxH": 0.20},
        {"bboxX": 0.60, "bboxY": 0.40, "bboxW": 0.25, "bboxH": 0.30},
    ]
    out = []
    for h in holds:
        bx1 = int(math.floor(h["bboxX"] * SYNTH_W))
        by1 = int(math.floor(h["bboxY"] * SYNTH_H))
        bx2 = int(math.ceil((h["bboxX"] + h["bboxW"]) * SYNTH_W))
        by2 = int(math.ceil((h["bboxY"] + h["bboxH"]) * SYNTH_H))
        w, hgt = bx2 - bx1, by2 - by1
        h["maskW"], h["maskH"] = w, hgt
        h["rle"] = [0, w * hgt]  # all-ones bbox mask
        lab = median_lab(rgba, h)
        out.append({"bboxX": h["bboxX"], "bboxY": h["bboxY"],
                    "bboxW": h["bboxW"], "bboxH": h["bboxH"],
                    "maskW": w, "maskH": hgt, "rle": h["rle"], "lab": lab})
    return {"imageWidth": SYNTH_W, "imageHeight": SYNTH_H, "holds": out}


# ------------------------------------------------------------------ export

def export() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)

    rgba = synth_rgba()
    (FIXTURES / "parity_synth_rgba.bin").write_bytes(rgba)
    tensor = reference_preprocess(rgba)
    (FIXTURES / "parity_preprocess_synth.bin").write_bytes(tensor.tobytes())

    ref = python_decode_reference()
    (FIXTURES / "parity_decode_reference.json").write_text(
        json.dumps(ref, separators=(",", ":")), encoding="utf-8"
    )

    lab = lab_reference(rgba)
    (FIXTURES / "parity_lab_reference.json").write_text(
        json.dumps(lab, separators=(",", ":")), encoding="utf-8"
    )

    print(f"decode holds: {len(ref['holds'])} on {ref['imageWidth']}x{ref['imageHeight']}")
    print(f"fixtures written to {FIXTURES}")


def selfcheck() -> None:
    # Deterministic pattern + RLE invariants
    rgba = synth_rgba()
    assert len(rgba) == SYNTH_W * SYNTH_H * 4
    assert rgba[0:4] == bytes((0, 0, 128, 255))          # gradient origin
    assert rle_encode(np.zeros((2, 2), np.uint8)) == [4]  # all zeros -> one run
    assert rle_encode(np.ones((2, 2), np.uint8)) == [0, 4]
    # Lab reference on the red block stays inside the block (bounded L range)
    lab = lab_reference(rgba)
    assert 20 < lab["holds"][0]["lab"]["l"] < 90
    print("selfcheck OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        export()
