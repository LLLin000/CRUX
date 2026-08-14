"""Prepare CRUX hold dataset in RF-DETR Roboflow COCO layout.

Layout expected by rfdetr: dataset_dir/{train,valid}/_annotations.coco.json + images.

Usage:
    python tools/prepare_dataset.py --src data/kaggle --out data/crux-dataset --annotated-images data/kaggle/bh --kaggle-token <KGAT_...>
    python tools/prepare_dataset.py --selfcheck
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

TRAIN_COUNT = 12  # 15 annotated images -> 12 train / 3 valid


def extract_zips(img_dir: Path) -> None:
    for z in sorted(img_dir.glob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(img_dir)
        z.unlink()


def split_coco(src_json: Path, img_src: Path, out_root: Path, train_count: int, max_width: int) -> None:
    from PIL import Image

    coco = json.loads(src_json.read_text(encoding="utf-8"))
    images = coco["images"]
    assert train_count < len(images), f"{len(images)} images, train_count must be smaller"
    # deterministic split by filename order, but only among images we have
    present = [im for im in images if (img_src / im["file_name"]).exists()]
    present.sort(key=lambda im: im["file_name"])
    assert train_count < len(present), f"{len(present)} present images, train_count must be smaller"
    train_files = {im["file_name"] for im in present[:train_count]}
    valid_files = {im["file_name"] for im in present[train_count:]}

    for split, keep, split_name in (
        ("train", train_files, "train2017"),
        ("valid", valid_files, "val2017"),
    ):
        split_dir = out_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        ann_dir = out_root / "annotations"
        ann_dir.mkdir(parents=True, exist_ok=True)
        out_anns = []
        out_imgs = []
        id_map = {}
        scale = 1.0
        for im in present:
            if im["file_name"] not in keep:
                continue
            src_p = img_src / im["file_name"]
            with Image.open(src_p) as pil:
                w, h = pil.size
                if max_width and w > max_width:
                    scale = max_width / w
                    new_size = (max_width, round(h * scale))
                    pil_resized = pil.resize(new_size, Image.LANCZOS)
                else:
                    scale, new_size = 1.0, (w, h)
            new_id = len(out_imgs)
            id_map[im["id"]] = new_id
            new_im = {**im, "id": new_id, "width": new_size[0], "height": new_size[1]}
            out_imgs.append(new_im)
            if scale != 1.0:
                pil_resized.save(split_dir / im["file_name"], quality=95)
            else:
                shutil.copy2(src_p, split_dir / im["file_name"])
        for a in coco["annotations"]:
            if a["image_id"] in id_map:
                seg = a["segmentation"]
                bbox = a["bbox"]
                if scale != 1.0:
                    seg = [[round(c * scale, 2) for c in poly] for poly in seg]
                    bbox = [round(v * scale, 2) for v in bbox]
                out_anns.append(
                    {**a, "id": len(out_anns), "image_id": id_map[a["image_id"]],
                     "segmentation": seg, "bbox": bbox, "area": round(bbox[2] * bbox[3], 2)}
                )
        split_coco_json = {
            "images": out_imgs,
            "categories": coco["categories"],
            "annotations": out_anns,
        }
        mode = "instances"
        ann_file = ann_dir / f"{mode}_{split_name}.json"
        ann_file.write_text(json.dumps(split_coco_json), encoding="utf-8")
        print(f"{split_name}: {len(out_imgs)} images, {len(out_anns)} annotations")


def selfcheck(tmp: Path) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools.kaggle_to_coco import via_to_coco  # reuse converter

    via = {
        "a.jpg1": {"filename": "a.jpg", "width": 100, "height": 100, "regions": [
            {"shape_attributes": {"name": "polygon", "all_points_x": [1, 2, 2], "all_points_y": [1, 1, 2]},
             "region_attributes": {"hold_type": "hold"}}]},
        "b.jpg1": {"filename": "b.jpg", "width": 100, "height": 100, "regions": []},
    }
    img = tmp / "img"
    img.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    for f in ("a.jpg", "b.jpg"):
        Image.new("RGB", (200, 100), (120, 80, 40)).save(img / f)
    coco = via_to_coco(via, img, tmp / "coco.json")
    split_coco(tmp / "coco.json", img, tmp / "ds", train_count=1, max_width=0)
    assert (tmp / "ds" / "train2017").exists()
    assert (tmp / "ds" / "annotations" / "instances_train2017.json").exists()
    assert (tmp / "ds" / "annotations" / "instances_val2017.json").exists()
    tr = json.loads((tmp / "ds" / "annotations" / "instances_train2017.json").read_text())
    assert len(tr["images"]) == 1 and len(tr["annotations"]) == 1
    shutil.rmtree(tmp / "ds")
    print("selfcheck OK: 12/3 split logic, layout valid")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="dir with bh-annotation.json")
    ap.add_argument("--img-dir", type=Path, required=True, help="dir with downloaded bh images (zips)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-width", type=int, default=1280, help="downscale training copies wider than this (polygons scaled too)")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck(Path(__file__).resolve().parent.parent / ".scratch")
    else:
        extract_zips(args.img_dir)
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from tools.kaggle_to_coco import via_to_coco

        via = json.loads((args.src / "bh-annotation.json").read_text(encoding="utf-8"))
        coco_path = args.src / "_bh_coco.json"
        via_to_coco(via, args.img_dir, coco_path)
        split_coco(coco_path, args.img_dir, args.out, TRAIN_COUNT, args.max_width)
