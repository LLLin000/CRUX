"""COCO (our pre-annotations) -> Label Studio task JSON with predictions.

LS PolygonLabels points are PERCENT of image size (0-100). Each COCO
annotation becomes one prediction polygon; predictions appear as pre-filled
regions the annotator can edit/delete.

Usage:
    python tools/coco_to_labelstudio.py \
        --coco data/preannotations/onnx_v101/preannotations.json \
        --out data/preannotations/labelstudio_tasks.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LABELS = {0: "hold", 1: "volume"}


def convert(coco: dict) -> list[dict]:
    anns_by_img: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    tasks = []
    for im in coco["images"]:
        w, h = im["width"], im["height"]
        results = []
        for a in anns_by_img.get(im["id"], []):
            pts = a["segmentation"][0]
            points = [[round(pts[i] / w * 100, 2), round(pts[i + 1] / h * 100, 2)]
                      for i in range(0, len(pts), 2)]
            label = LABELS.get(a.get("category_id", 0), "hold")
            results.append({
                "from_name": "label", "to_name": "image",
                "type": "polygonlabels",
                "value": {"points": points, "polygonlabels": [label]},
            })
        task = {"data": {"image": f"/data/upload/{im['file_name']}"}}
        if results:
            task["predictions"] = [{
                "result": results,
                "score": 1.0,
                "model_version": "crux-v0.3.0-preannotate",
            }]
        tasks.append(task)
    return tasks


def main(coco_path: Path, out_path: Path) -> None:
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    tasks = convert(coco)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path} ({len(tasks)} tasks, "
          f"{sum(1 for t in tasks if 'predictions' in t)} with predictions)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", type=Path,
                    default=Path("data/preannotations/onnx_v101/preannotations.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/preannotations/labelstudio_tasks.json"))
    args = ap.parse_args()
    main(args.coco, args.out)
