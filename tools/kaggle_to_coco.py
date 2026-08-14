"""Kaggle VIA JSON -> COCO JSON converter for CRUX D0.

Input:  VIA project JSON (as in xiaoxiae/Indoor-Climbing-Hold-and-Route-Segmentation Kaggle dataset)
        region_attributes: {label_type: handlabeled, hold_type: hold|volume}
Output: COCO JSON consumable by RF-DETR training.

Usage:
    python tools/kaggle_to_coco.py <via.json> <images_dir> <out.json>
    python tools/kaggle_to_coco.py --selfcheck

License discipline: this is a data-processing tool, no model code. Kaggle data
remains research-only per PLAN §2 rule 2.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CATEGORIES = [{"id": 0, "name": "hold"}, {"id": 1, "name": "volume"}]


def via_to_coco(via: dict, images_dir: Path, out: Path) -> dict:
    coco = {"images": [], "categories": CATEGORIES, "annotations": []}
    img_id = ann_id = 0
    meta = via.get("_via_img_metadata", via)  # VIA exports metadata under this key

    for key, img_meta in meta.items():
        if key.startswith("_"):
            continue
        filename = img_meta.get("filename", key)
        size = img_meta.get("size", 0)
        width = img_meta.get("width") or 0
        height = img_meta.get("height") or 0
        if not (width and height):
            # VIA export sometimes omits dims; fall back to image file probe.
            from PIL import Image  # noqa: PLC0415

            p = images_dir / filename
            if p.exists():
                with Image.open(p) as im:
                    width, height = im.size
        coco["images"].append(
            {"id": img_id, "file_name": filename, "width": width, "height": height}
        )

        for region in img_meta.get("regions", []):
            attrs = region.get("region_attributes", {})
            hold_type = attrs.get("hold_type", "hold")
            if hold_type not in ("hold", "volume"):
                continue
            shape = region.get("shape_attributes", {})
            if shape.get("name") != "polygon":
                continue  # VIA ellipse/rect regions not used in this dataset
            xs = shape.get("all_points_x", [])
            ys = shape.get("all_points_y", [])
            if len(xs) < 3 or len(xs) != len(ys):
                continue

            poly = [c for xy in zip(xs, ys) for c in xy]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            area = shoelace(xs, ys)
            if area <= 0:
                continue
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 0 if hold_type == "hold" else 1,
                    "segmentation": [poly],
                    "area": round(area, 2),
                    "bbox": [min_x, min_y, max_x - min_x, max_y - min_y],
                    "iscrowd": 0,
                }
            )
            ann_id += 1
        img_id += 1

    out.write_text(json.dumps(coco), encoding="utf-8")
    return coco


def shoelace(xs: list, ys: list) -> float:
    n = len(xs)
    return abs(sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i] for i in range(n))) / 2.0


def selfcheck() -> None:
    via = {
        "0000.jpg0": {
            "filename": "0000.jpg",
            "size": 1,
            "width": 100,
            "height": 200,
            "regions": [
                {"shape_attributes": {"name": "polygon",
                                      "all_points_x": [10, 40, 40, 10],
                                      "all_points_y": [10, 10, 30, 30]},
                 "region_attributes": {"hold_type": "hold"}},
                {"shape_attributes": {"name": "polygon",
                                      "all_points_x": [50, 80, 65],
                                      "all_points_y": [50, 50, 80]},
                 "region_attributes": {"hold_type": "volume"}},
                {"shape_attributes": {"name": "rect", "x": 0, "y": 0},
                 "region_attributes": {"hold_type": "hold"}},  # skipped
                {"shape_attributes": {"name": "polygon",
                                      "all_points_x": [1, 2, 3],
                                      "all_points_y": [1, 2, 3]},  # zero area, skipped
                 "region_attributes": {"hold_type": "hold"}},
            ],
        },
        "_via_settings": {},
    }
    out = Path("__selftest_coco.json")
    coco = via_to_coco(via, Path("."), out)
    anns = coco["annotations"]
    assert len(coco["images"]) == 1, coco["images"]
    assert len(anns) == 2, f"expected 2 valid anns, got {len(anns)}"
    assert anns[0]["category_id"] == 0 and anns[0]["bbox"] == [10, 10, 30, 20]
    assert anns[1]["category_id"] == 1 and abs(anns[1]["area"] - 450.0) < 1e-6
    assert anns[0]["segmentation"][0] == [10, 10, 40, 10, 40, 30, 10, 30]
    out.unlink()
    print("selfcheck OK: 2 valid instances (hold/volume), rect+zero-area skipped")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selfcheck":
        selfcheck()
    elif len(sys.argv) == 4:
        via = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        coco = via_to_coco(via, Path(sys.argv[2]), Path(sys.argv[3]))
        n_img = len(coco["images"])
        n_ann = len(coco["annotations"])
        print(f"OK: {n_img} images, {n_ann} annotations -> {sys.argv[3]}")
    else:
        print(__doc__)
        sys.exit(2)
