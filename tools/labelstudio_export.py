"""Export Label Studio annotations (even drafts) -> our COCO format.

LS polygon points are PERCENT (0-100) of image size; labels map hold/volume
to category ids 0/1. Output merges into data/crux-dataset style JSON.

Usage:
    python tools/labelstudio_export.py --project 1 --out data/preannotations/annotated_coco.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import requests

BASE = "http://localhost:8080"
EMAIL = "admin@localhost"
PASSWORD = "crux-d1-admin"
LABEL_IDS = {"hold": 0, "volume": 1}


def login() -> requests.Session:
    s = requests.Session()
    s.get(f"{BASE}/user/login")
    csrf = s.cookies.get("csrftoken", "")
    s.post(f"{BASE}/user/login",
           data={"email": EMAIL, "password": PASSWORD,
                 "csrfmiddlewaretoken": csrf},
           headers={"Referer": f"{BASE}/user/login"})
    return s


def export(project: int, out_path: Path, photos_dir: Path) -> None:
    s = login()
    d = s.get(f"{BASE}/api/tasks?project={project}").json()
    tasks = d["tasks"]

    images, annotations = [], []
    ann_id = 0
    for t in tasks:
        filename = t["data"]["image"].split("/")[-1]
        img = cv2.imread(str(photos_dir / filename))
        if img is None:
            print(f"!! missing {filename}"); continue
        h, w = img.shape[:2]
        img_id = len(images)
        images.append({"id": img_id, "file_name": filename,
                       "width": w, "height": h})

        detail = s.get(f"{BASE}/api/tasks/{t['id']}/").json()
        anns = detail.get("annotations") or []
        for a in anns:
            for r in a.get("result") or []:
                if r.get("type") != "polygonlabels":
                    continue
                label = (r.get("value", {}).get("polygonlabels") or ["hold"])[0]
                cat = LABEL_IDS.get(label, 0)
                pts_pct = r["value"]["points"]  # [[x%, y%], ...]
                pts = [[p[0] / 100 * w, p[1] / 100 * h] for p in pts_pct]
                if len(pts) < 3:
                    continue
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                import numpy as np
                annotations.append({
                    "id": ann_id, "image_id": img_id, "category_id": cat,
                    "segmentation": [[float(v) for xy in pts for v in xy]],
                    "area": float(cv2.contourArea(np.array(pts, dtype="float32"))),
                    "bbox": [float(min(xs)), float(min(ys)),
                             float(max(xs) - min(xs)), float(max(ys) - min(ys))],
                    "iscrowd": 0,
                })
                ann_id += 1
        print(f"{filename}: {sum(1 for a in annotations if a['image_id'] == img_id)} anns")

    coco = {
        "images": images,
        "categories": [{"id": 0, "name": "hold"}, {"id": 1, "name": "volume"}],
        "annotations": annotations,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(__import__("json").dumps(coco), encoding="utf-8")
    print(f"wrote {out_path}: {len(images)} images, {len(annotations)} annotations")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=int, default=1)
    ap.add_argument("--out", type=Path,
                    default=Path("data/preannotations/annotated_coco.json"))
    ap.add_argument("--photos", type=Path, default=Path("data/realpic"))
    args = ap.parse_args()
    export(args.project, args.out, args.photos)
