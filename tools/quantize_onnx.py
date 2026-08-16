"""Quantize a CRUX ONNX model to INT8 (static, QDQ) with calibration.

FP32 source -> INT8 (~4x smaller). Calibration runs the FP32 model over
representative images (train set) collecting activation ranges.

Usage:
    python tools/quantize_onnx.py \
        --model output/onnx_v101/crux-hold-seg-v1.0.1-648-fp32.onnx \
        --out output/onnx_v101/crux-hold-seg-v1.0.1-648-int8.onnx \
        --calib data/crux-dataset/train2017 --resolution 648
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


class CalibReader:
    """Yields preprocessed images from a directory as calibration batches."""

    def __init__(self, img_dir: Path, input_name: str, resolution: int, max_imgs: int = 20):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from infer_real import preprocess
        from PIL import Image

        self._preprocess = preprocess
        self._Image = Image
        self._input_name = input_name
        self._res = resolution
        self._paths = sorted(img_dir.glob("*.jpg"))[:max_imgs]
        self._it = iter(self._paths)

    def get_next(self) -> dict | None:
        """onnxruntime CalibrationDataReader protocol: one sample or None."""
        try:
            p = next(self._it)
        except StopIteration:
            return None
        pil = self._Image.open(p).convert("RGB")
        return {self._input_name: self._preprocess(pil, self._res)}

    def get_calibration_data(self):
        return self


def main(model_path: Path, out_path: Path, calib_dir: Path, resolution: int) -> None:
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    import onnxruntime as ort

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    reader = CalibReader(calib_dir, input_name, resolution)
    quantize_static(
        str(model_path), str(out_path), reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        extra_options={"MinimumRealRange": 0.0001},
    )
    print(f"quantized: {out_path}")

    # sanity: input shape + size
    s2 = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    print(f"  input: {s2.get_inputs()[0].shape}  size: {out_path.stat().st_size/2**20:.0f} MB "
          f"(orig {model_path.stat().st_size/2**20:.0f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True, help="FP32 ONNX")
    ap.add_argument("--out", type=Path, required=True, help="INT8 output path")
    ap.add_argument("--calib", type=Path,
                    default=Path("data/crux-dataset/train2017"))
    ap.add_argument("--resolution", type=int, default=648)
    args = ap.parse_args()
    main(args.model, args.out, args.calib, args.resolution)
