"""Infer the MaskRCNN+MobileNetV3 baseline on real photos for visual comparison
against RF-DETR-Seg-S. Same resize convention as training (short side 640)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np
import torch
from PIL import Image

from train_hold_maskrcnn import build_model


def main(photos: Path, checkpoint: Path, out_dir: Path, min_size: int, conf: float) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.to(device).eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(photos.glob("*.jpg")):
        pil = Image.open(f).convert("RGB")
        w, h = pil.size
        scale = min_size / min(h, w)
        img = cv2.resize(np.array(pil), (int(w * scale), int(h * scale)))
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            pred = model([t.to(device)])[0]
        boxes = pred["boxes"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()
        masks = pred["masks"].cpu().numpy()  # (N,1,H,W)

        vis = img.copy()
        n_hold = n_vol = 0
        for i in range(len(boxes)):
            if scores[i] < conf:
                continue
            cls = int(labels[i]) - 1  # 0=hold, 1=volume
            color = (0, 200, 255) if cls == 0 else (255, 80, 255)
            m = masks[i, 0] > 0.5
            overlay = vis.copy()
            overlay[m] = overlay[m] * 0.5 + np.array(color) * 0.5
            vis = cv2.addWeighted(overlay, 0.7, vis, 0.3, 0)
            x1, y1, x2, y2 = [int(v) for v in boxes[i]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"{scores[i]:.2f}", (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            if cls == 0:
                n_hold += 1
            else:
                n_vol += 1
        dest = out_dir / f"{f.stem}_annotated.jpg"
        cv2.imwrite(str(dest), vis)
        print(f"{f.name}: {n_hold} holds, {n_vol} volumes (conf>={conf})")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", type=Path, default=Path("data/realpic"))
    ap.add_argument("--checkpoint", type=Path, default=Path("output/maskrcnn_mobilenet/best_model.pth"))
    ap.add_argument("--out", type=Path, default=Path("data/realpic_out/maskrcnn"))
    ap.add_argument("--resolution", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.3)
    args = ap.parse_args()
    main(args.photos, args.checkpoint, args.out, args.resolution, args.conf)
