"""Interactive pre-annotation corrector for D1 (v0.3.0 pre-annotations).

Loads a COCO file with model pre-annotations, shows each photo with the
polygons overlaid, and lets a human correct them:
  click polygon  -> select (red);  X / Delete -> remove (false positive)
  click empty    -> new circular hold; drag to size it; Enter confirms
  N / P          -> next / previous image
  S              -> save corrections (same COCO file)
  Q / Esc        -> quit (saves)

Usage:
    python tools/annotate.py --coco data/preannotations/onnx_v101/preannotations.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

SELECT = (60, 60, 255)     # BGR red
HOLD = (255, 200, 60)      # BGR amber
NEW = (80, 255, 120)       # BGR green


class Annotator:
    def __init__(self, coco_path: Path):
        self.coco_path = coco_path
        self.coco = json.loads(coco_path.read_text(encoding="utf-8"))
        # data/preannotations/<tag>/xxx.json -> data/realpic (fallback: repo root)
        self.img_dir = coco_path.parent.parent.parent / "realpic"
        if not self.img_dir.exists():
            self.img_dir = Path("data/realpic")
        self.anns_by_img: dict[int, list[dict]] = {}
        for a in self.coco["annotations"]:
            self.anns_by_img.setdefault(a["image_id"], []).append(a)
        self.idx = 0
        self.selected = None          # annotation index within current image
        self.drawing = False
        self.draw_center: tuple[int, int] | None = None
        self.draw_radius = 0

    def current_image(self) -> dict:
        return self.coco["images"][self.idx]

    def show(self) -> None:
        im = self.current_image()
        img = cv2.imread(str(self.img_dir / im["file_name"]))
        if img is None:
            print(f"!! missing {im['file_name']}"); return
        anns = self.anns_by_img.get(im["id"], [])
        for i, a in enumerate(anns):
            pts = np.array(a["segmentation"][0]).reshape(-1, 2).astype(np.int32)
            color = SELECT if i == self.selected else HOLD
            cv2.polylines(img, [pts], True, color, 2)
            if i == self.selected:
                cv2.drawContours(img, [pts], -1, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        if self.draw_center and self.drawing:
            cv2.circle(img, self.draw_center, self.draw_radius, NEW, 2)
        cv2.putText(img, f"{self.idx + 1}/{len(self.coco['images'])}  {im['file_name']}  "
                         f"[{len(anns)} holds]  N/P next  X del  Click draw  S save  Q quit",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("CRUX annotate", img)

    def run(self) -> None:
        cv2.namedWindow("CRUX annotate")
        cv2.setMouseCallback("CRUX annotate", lambda *a: mouse_cb(*a, self))
        self.show()
        while True:
            k = cv2.waitKey(20) & 0xFF
            if k == ord("q") or k == 27:
                break
            if k in (ord("n"), 32):
                self.idx = min(len(self.coco["images"]) - 1, self.idx + 1)
                self.selected = None; self.drawing = False
                self.show()
            elif k == ord("p") or k == 8:
                self.idx = max(0, self.idx - 1)
                self.selected = None; self.drawing = False
                self.show()
            elif k in (ord("x"), ord("d"), 127):
                self.delete_selected()
            elif k == ord("s"):
                self.save()
            elif k == 13 and self.drawing and self.draw_center:  # Enter confirms new hold
                self.add_circle_hold()
        cv2.destroyAllWindows()

    def delete_selected(self) -> None:
        im = self.current_image()
        anns = self.anns_by_img.get(im["id"], [])
        if self.selected is None or not (0 <= self.selected < len(anns)):
            return
        a = anns[self.selected]
        self.coco["annotations"].remove(a)
        del self.anns_by_img[im["id"]][self.selected]
        self.selected = None
        self.show()

    def add_circle_hold(self) -> None:
        im = self.current_image()
        cx, cy = self.draw_center
        r = self.draw_radius
        if r < 4:
            self.drawing = False; return
        seg = []
        for deg in range(0, 360, 12):
            a = np.deg2rad(deg)
            seg += [float(cx + r * np.cos(a)), float(cy + r * np.sin(a))]
        self.coco["annotations"].append({
            "id": max((a["id"] for a in self.coco["annotations"]), default=-1) + 1,
            "image_id": im["id"], "category_id": 0,
            "segmentation": [seg],
            "area": float(np.pi * r * r),
            "bbox": [float(cx - r), float(cy - r), float(2 * r), float(2 * r)],
            "iscrowd": 0, "preannotated": False, "confidence": 1.0,
        })
        self.anns_by_img.setdefault(im["id"], []).append(self.coco["annotations"][-1])
        self.drawing = False; self.draw_center = None
        self.show()

    def save(self) -> None:
        # drop the temporary "drawing" annotation if any
        self.coco_path.write_text(json.dumps(self.coco), encoding="utf-8")
        print(f"saved {self.coco_path} ({len(self.coco['annotations'])} anns)")


def mouse_cb(event, x, y, flags, ann: Annotator):
    if event == cv2.EVENT_LBUTTONDOWN:
        im = ann.current_image()
        anns = ann.anns_by_img.get(im["id"], [])
        hit = None
        for i, a in enumerate(anns):
            pts = np.array(a["segmentation"][0]).reshape(-1, 2)
            if cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0:
                hit = i; break
        if hit is not None:
            ann.selected = hit
            ann.drawing = False
        else:
            ann.selected = None
            ann.drawing = True
            ann.draw_center = (x, y)
            ann.draw_radius = 0
        ann.show()
    elif event == cv2.EVENT_MOUSEMOVE and ann.drawing and ann.draw_center:
        ann.draw_radius = int(np.hypot(x - ann.draw_center[0], y - ann.draw_center[1]))
        ann.show()


def main(coco_path: Path) -> None:
    ann = Annotator(coco_path)
    ann.run()
    print("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", type=Path,
                    default=Path("data/preannotations/onnx_v101/preannotations.json"))
    args = ap.parse_args()
    main(args.coco)
