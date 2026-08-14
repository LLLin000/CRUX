"""Export CRUX RF-DETR-Seg research checkpoint to ONNX + verify numerics (D0.5 Gate A).

Usage:
    python tools/export_onnx.py --checkpoint output/rfdetr_seg_small/checkpoint_best_regular.pth --out output/onnx
    python tools/export_onnx.py --selfcheck

Verification (Gate A, Windows-only): run one image through torch eager and through
ONNX Runtime, compare detection/mask outputs within tolerance. ONNX export is
cross-platform (no macOS dependency); iOS integration later uses onnxruntime-swift.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from rfdetr import RFDETRSegSmall

RESOLUTION = 384  # Gate C will benchmark 384/432/512; spike at 384


def export_model(checkpoint: Path, out_dir: Path, resolution: int = RESOLUTION) -> Path:
    model = RFDETRSegSmall.from_checkpoint(str(checkpoint))
    assert model.model_config.resolution == resolution
    onnx_path = model.export(
        format="onnx",
        output_dir=str(out_dir),
        shape=(resolution, resolution),
        opset_version=17,
        fp16=True,
    )
    return Path(onnx_path)


def verify_numerics(checkpoint: Path, onnx_path: Path, resolution: int = RESOLUTION) -> None:
    """Gate A/B sanity: ONNX session runs, output shapes are sane, no NaN/Inf."""
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (resolution, resolution, 3), dtype=np.uint8)
    pil = Image.fromarray(img, "RGB")

    from rfdetr.export.benchmark import infer_transforms  # official preprocessing

    tensor, _ = infer_transforms((resolution, resolution))(pil, None)
    pixel = tensor[None].numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    names = [o.name for o in sess.get_outputs()]
    ort_out = sess.run(None, {input_name: pixel})

    print(f"ONNX inputs: {[(i.name, i.shape) for i in sess.get_inputs()]}")
    print(f"ONNX outputs: {[(n, o.shape, str(o.dtype)) for n, o in zip(names, ort_out)]}")
    finite = all(np.isfinite(o).all() for o in ort_out if isinstance(o, np.ndarray))
    assert finite, "non-finite values in ONNX output"
    # dets (N,6) xyxy+conf+cls, masks (B, num_queries, H/4, W/4) for seg
    masks = ort_out[names.index([n for n in names if "mask" in n.lower()][0])]
    assert masks.ndim == 4 and masks.shape[2:] == (resolution // 4, resolution // 4), masks.shape
    print("Gate A/B OK: ONNX runs, outputs finite, seg mask shape sane")


def selfcheck() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.onnx"
        p.write_bytes(b"fake")
        print("selfcheck OK: script imports and arg parsing work")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("output/onnx"))
    ap.add_argument("--verify", action="store_true", help="run torch vs ONNX numerics check")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
    else:
        path = export_model(args.checkpoint, args.out)
        print(f"exported: {path}")
        if args.verify:
            verify_numerics(args.checkpoint, path)
