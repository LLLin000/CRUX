"""Small-model baseline: torchvision Mask R-CNN with MobileNetV3-Large FPN
backbone (~10-15M params, BSD-3) vs RF-DETR-Seg-S (33.7M, Apache-2.0).

RTMDet-Ins-tiny (5.6M) is the ideal candidate but mmcv has no Windows
wheels for Python 3.14 — this torchvision build is the environment-viable
small model for the hold task. Same data, same resolution (640), same COCO
format as the Seg-S runs, so mAP is directly comparable.

Usage:
    python tools/train_hold_maskrcnn.py --epochs 100 --resolution 640
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN, LastLevelMaxPool
from torchvision.ops import MultiScaleRoIAlign

OUTPUT_DIR = Path("output/maskrcnn_mobilenet")
DATA_DIR = Path("data/crux-dataset")


def mobilenet_fpn_backbone() -> BackboneWithFPN:
    """MobileNetV3-Large features sliced into C2-C5 (strides 4/8/16/32) + FPN.
    ~5.5M backbone params; the Mask R-CNN head adds ROI heads."""
    from torchvision.models import mobilenet_v3_large
    from torchvision.models.detection.backbone_utils import IntermediateLayerGetter
    m = mobilenet_v3_large(weights="IMAGENET1K_V2")
    f = m.features
    # C2:  stride 4,  out 24   (features 0..3)
    # C3:  stride 8,  out 40   (features 4..6)
    # C4:  stride 16, out 112  (features 7..12)
    # C5:  stride 32, out 160  (features 13..15; drop the 1x1 960 conv)

    class _Body(nn.Module):
        def __init__(self, c2, c3, c4, c5):
            super().__init__()
            self.c2, self.c3, self.c4, self.c5 = c2, c3, c4, c5

    body = _Body(f[0:4], f[4:7], f[7:13], f[13:16])
    backbone = IntermediateLayerGetter(body, {"c2": "0", "c3": "1", "c4": "2", "c5": "3"})
    return BackboneWithFPN(
        backbone,
        return_layers={"c2": "0", "c3": "1", "c4": "2", "c5": "3"},
        in_channels_list=[24, 40, 112, 160],
        out_channels=256,
        extra_blocks=LastLevelMaxPool(),
    )


def build_model(num_classes: int = 3) -> MaskRCNN:
    backbone = mobilenet_fpn_backbone()
    model = MaskRCNN(
        backbone,
        num_classes=num_classes,  # 0=bg, 1=hold, 2=volume
        min_size=640, max_size=1333,
        rpn_pre_nms_top_n_train=2000, rpn_post_nms_top_n_train=1000,
    )
    return model


class CocoDetection(torch.utils.data.Dataset):
    """COCO json + image dir, returns (img_tensor, target_dict)."""

    def __init__(self, ann_file: Path, img_dir: Path, min_size: int = 640):
        import cv2
        self.cv2 = cv2
        self.img_dir = img_dir
        self.min_size = min_size
        data = json.loads(ann_file.read_text(encoding="utf-8"))
        self.imgs = {im["id"]: im for im in data["images"]}
        anns = {}
        for a in data["annotations"]:
            anns.setdefault(a["image_id"], []).append(a)
        self.anns = {iid: v for iid, v in anns.items() if iid in self.imgs}
        self.ids = sorted(self.imgs.keys())
        assert all(c["id"] in (0, 1) for c in data["categories"]), data["categories"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        iid = self.ids[idx]
        im = self.imgs[iid]
        img = self.cv2.imread(str(self.img_dir / im["file_name"]))
        img = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        # resize so the SHORT side == min_size (keep aspect)
        scale = self.min_size / min(h, w)
        img = self.cv2.resize(img, (int(w * scale), int(h * scale)))
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        boxes, masks, labels = [], [], []
        for a in self.anns.get(iid, []):
            x, y, bw, bh = a["bbox"]
            x2, y2 = x + bw, y + bh
            if bw < 1 or bh < 1:
                continue
            boxes.append([x * scale, y * scale, x2 * scale, y2 * scale])
            labels.append(a["category_id"] + 1)  # 0=bg, 1=hold, 2=volume
            import pycocotools.mask as mask_util
            seg = a["segmentation"]
            if isinstance(seg, list):  # polygon (flat or nested)
                rle = mask_util.frPyObjects(seg, h, w)
                if isinstance(rle, list):
                    rle = mask_util.merge(rle)
                m = mask_util.decode(rle)
            else:  # RLE dict
                m = mask_util.decode(seg)
            if m.ndim == 3:  # RLE per category? take max over classes
                m = m.max(axis=2)
            m = self.cv2.resize(m.astype("uint8"), (int(w * scale), int(h * scale)))
            masks.append(m > 0.5)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "labels": torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "masks": torch.as_tensor(np_stack(masks), dtype=torch.uint8) if masks else torch.zeros((0, h, w), dtype=torch.uint8),
            "image_id": torch.tensor([iid]),
            "area": torch.as_tensor([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes], dtype=torch.float32) if boxes else torch.zeros((0,)),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        return img, target


def np_stack(masks):
    import numpy as np
    if not masks:
        return np.zeros((0, 0, 0), np.uint8)
    return np.stack(masks)


def collate(batch):
    return tuple(zip(*batch))


def train_one_epoch(model, loader, opt, device, epoch):
    model.train()
    total = 0
    for i, (imgs, targets) in enumerate(loader):
        imgs = [im.to(device) for im in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(imgs, targets)
        loss = sum(l for l in loss_dict.values())
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += float(loss)
        if i % 5 == 0:
            print(f"  epoch {epoch} step {i}: loss={float(loss):.3f} " +
                  " ".join(f"{k}={float(v):.3f}" for k, v in loss_dict.items()))
    return total / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device):
    import numpy as np
    import pycocotools.mask as mask_util
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    model.eval()
    results, anns_all = [], []
    img_meta = {}
    for i, (imgs, targets) in enumerate(loader):
        for im, t in zip(imgs, targets):
            img_meta[int(t["image_id"][0])] = (im.shape[1], im.shape[2])
            pred = model([im.to(device)])[0]
            im_h, im_w = im.shape[1:]
            for bi in range(len(pred["boxes"])):
                x1, y1, x2, y2 = [float(v) for v in pred["boxes"][bi].cpu()]
                sc = float(pred["scores"][bi].cpu())
                cl = int(pred["labels"][bi].cpu())
                mask = (pred["masks"][bi, 0].cpu().numpy() > 0.5)
                rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
                results.append({
                    "image_id": int(t["image_id"][0]),
                    "category_id": cl - 1,
                    "score": sc,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "segmentation": rle,
                })
            # ground truth for eval
            for bi in range(len(t["boxes"])):
                x1, y1, x2, y2 = [float(v) for v in t["boxes"][bi].cpu()]
                m = t["masks"][bi].cpu().numpy()
                rle = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))
                anns_all.append({
                    "image_id": int(t["image_id"][0]),
                    "category_id": int(t["labels"][bi].cpu()) - 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": float((x2 - x1) * (y2 - y1)),
                    "segmentation": rle, "iscrowd": 0, "id": len(anns_all),
                })
    # COCO eval
    if not results or not anns_all:
        return 0.0, 0.0  # untrained model yields no detections yet
    coco_gt = COCO()
    coco_gt.dataset = {
        "images": [{"id": i, "height": h_img, "width": w_img} for i, (h_img, w_img) in img_meta.items()],
        "categories": [{"id": 0, "name": "hold"}, {"id": 1, "name": "volume"}],
        "annotations": anns_all,
    }
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, coco_dt, "segm")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return ev.stats[0], ev.stats[1]  # mAP_50_95, mAP_50


def main(epochs: int, resolution: int, output_dir: Path) -> None:
    import numpy as np  # noqa
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    model = build_model()
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MaskRCNN+MobileNetV3 params: {n_params / 1e6:.1f}M")

    train_ds = CocoDetection(DATA_DIR / "annotations" / "instances_train2017.json",
                             DATA_DIR / "train2017", min_size=resolution)
    val_ds = CocoDetection(DATA_DIR / "annotations" / "instances_val2017.json",
                           DATA_DIR / "val2017", min_size=resolution)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=2, shuffle=True,
                                               collate_fn=collate, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False,
                                             collate_fn=collate, num_workers=0)

    opt = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(epochs * 0.4), int(epochs * 0.7)], gamma=0.1)

    output_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for ep in range(1, epochs + 1):
        avg = train_one_epoch(model, train_loader, opt, device, ep)
        sched.step()
        print(f"[epoch {ep}] train loss {avg:.3f}")
        if ep % 10 == 0 or ep == epochs:
            print(f"  val (epoch {ep}):")
            mAP, mAP50 = evaluate(model, val_loader, device)
            if mAP > best:
                best = mAP
                torch.save(model.state_dict(), output_dir / "best_model.pth")
                print(f"  saved best (mAP_50_95={mAP:.4f})")
        if ep % 10 == 0:
            torch.save(model.state_dict(), output_dir / f"checkpoint_{ep}.pth")
    print(f"done. best mAP_50_95={best:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--resolution", type=int, default=640)
    ap.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = ap.parse_args()
    main(args.epochs, args.resolution, Path(args.output_dir))
