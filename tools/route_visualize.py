"""Visual experiments: color-match route rendering on real photo annotations.

Pipeline: ONNX model detections -> per-hold Lab median -> seed color (blue)
-> dE00 color match -> DBSCAN grouping -> three renderings:
  1. highlight: matched holds popped, everything else dimmed
  2. badge: cartoonish flat-color route (chunky outline, posterized bg)
  3. route line: connects matched holds (group order) + cluster stats

Usage:
    python tools/route_visualize.py --photo data/realpic/66ced02810271ec75cddbac23909f407.jpg \
        --model output/onnx_640_aug/rfdetr-seg-small-aug.onnx --seed-lab "35 5 -25" --out data/route_viz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np
from PIL import Image

from infer_real import preprocess, decode, mask_to_polygon
from route_selector import Hold, SeededRouteSelector

# blue volume/route color target (Lab); override with --seed-lab
BLUE_LAB = np.array([35.0, 5.0, -25.0])


def detect_holds(sess, photo: Path, inp_name: str, res: int) -> list[Hold]:
    pil = Image.open(photo).convert("RGB")
    w, h = pil.size
    out = sess.run(None, {inp_name: preprocess(pil, res)})
    boxes, confs, cls_ids, masks = decode(out, w, h, res)
    holds = []
    for i, (box, conf, cls, m) in enumerate(zip(boxes, confs, cls_ids, masks)):
        if cls != 0:  # holds only for route color matching
            continue
        ys, xs = np.nonzero(m)
        if len(xs) < 50:
            continue
        mask = m > 0.5
        holds.append(Hold(id=i, mask=mask, centroid=(float(xs.mean()), float(ys.mean()))))
    return holds


def lab_of_hold(img_rgb: np.ndarray, hold: Hold) -> np.ndarray:
    from skimage.color import rgb2lab
    lab = rgb2lab(img_rgb)
    px = lab[hold.mask]
    return np.median(px, axis=0)


def pick_blue_seed(holds_lab: dict[int, np.ndarray], target: np.ndarray) -> int:
    from skimage.color import deltaE_ciede2000
    best, best_d = None, 1e9
    for hid, lab in holds_lab.items():
        d = deltaE_ciede2000(target[None], lab[None])[0]
        if d < best_d:
            best, best_d = hid, d
    return best


def main(photo: Path, model_path: Path, seed_lab: np.ndarray, out_dir: Path) -> None:
    import onnxruntime as ort
    from infer_real import session_resolution

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    res = session_resolution(sess)
    holds = detect_holds(sess, photo, sess.get_inputs()[0].name, res)

    img_rgb = np.array(Image.open(photo).convert("RGB"))
    holds_lab = {h.id: lab_of_hold(img_rgb, h) for h in holds}
    seed_id = pick_blue_seed(holds_lab, seed_lab)
    print(f"{len(holds)} holds; blue seed hold #{seed_id} "
          f"lab={np.round(holds_lab[seed_id], 1)} (target {seed_lab})")

    sel = SeededRouteSelector(img_rgb, holds, deltaE_threshold=12.0)
    matched = sel.select(seed_id)
    print(f"dE00<12 matched {len(matched)} holds: ids={sorted(h.id for h in matched)}")

    # DBSCAN spatial groups among matched (eps = 3x median diameter)
    matched_ids = sorted(h.id for h in matched)
    centers = np.array([holds_lab and holds[hid].centroid for hid in matched_ids]) if False else \
        np.array([next(h.centroid for h in holds if h.id == hid) for hid in matched_ids])
    if len(centers) > 1:
        from sklearn.cluster import DBSCAN
        diam = sel._median_diameter()
        db = DBSCAN(eps=3.0 * diam, min_samples=1).fit(centers)
        groups = {}
        for hid, lbl in zip(matched_ids, db.labels_):
            groups.setdefault(int(lbl), []).append(hid)
        print(f"DBSCAN groups: {[sorted(g) for g in groups.values()]}")

    # ---------- rendering 1: highlight ----------
    hi = np.array(img_rgb)
    for h in holds:
        if h.id not in matched_ids:
            hi[h.mask] = (hi[h.mask] * 0.25).astype(np.uint8)
    for hid in matched_ids:
        h = next(x for x in holds if x.id == hid)
        hi[h.mask] = np.maximum(hi[h.mask], np.array([120, 200, 255]))  # pop blue-ish
        outline(hi, h, (255, 255, 255), 3)
    cv2.imwrite(str(out_dir / "1_highlight.jpg"), cv2.cvtColor(hi, cv2.COLOR_RGB2BGR))

    # ---------- rendering 2: badge (cartoon) ----------
    badge = cartoon(img_rgb, holds, matched_ids)
    cv2.imwrite(str(out_dir / "2_badge.jpg"), cv2.cvtColor(badge, cv2.COLOR_RGB2BGR))

    # ---------- rendering 3: route line ----------
    route = np.array(img_rgb)
    order = route_order(holds, matched_ids)
    pts = np.array([holds[hid].centroid for hid in order], np.int32)
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(route, tuple(a), tuple(b), (255, 120, 60), 6, cv2.LINE_AA)
    for hid in order:
        x, y = holds[hid].centroid
        cv2.circle(route, (int(x), int(y)), 12, (255, 255, 255), -1)
        cv2.circle(route, (int(x), int(y)), 6, (60, 120, 255), -1)
    cv2.imwrite(str(out_dir / "3_route_line.jpg"), cv2.cvtColor(route, cv2.COLOR_RGB2BGR))

    print(f"wrote {out_dir}/1_highlight.jpg, 2_badge.jpg, 3_route_line.jpg")


def outline(img: np.ndarray, hold: Hold, color, thickness: int) -> None:
    """Smooth outline via the shared edge-preserving two-stage RDP polygon
    (same path as pre-annotations) — never the raw raster mask edge."""
    poly = mask_to_polygon(hold.mask)
    if poly is None:
        m = (hold.mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, color, thickness)
        return
    pts = np.array(poly[0]).reshape(-1, 2).astype(np.int32)
    cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)


def cartoon(img_rgb: np.ndarray, holds: list[Hold], matched_ids: list[int]) -> np.ndarray:
    """Badge-style cartoon, deep ink-blue backdrop:
    - backdrop: fixed navy radial gradient (independent of the photo's murky
      dominant color — dark walls give meaningless palette), soft center glow
    - ALL holds keep posterized surface detail (L 6 levels, ab 8 levels so
      hues don't smear) stamped as stickers, slightly lifted so they read
    - matched route: white outline + modest saturation bump (1.25, not 1.6)
    - edges: light ink (dark backdrop needs light lines)"""
    h, w = img_rgb.shape[:2]

    # --- fixed navy gradient backdrop ---
    c_hi = np.array([46, 62, 87])     # #2e3e57 mid navy
    c_deep = np.array([12, 18, 30])   # #0c121e near-black blue
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w * 0.5, h * 0.42
    r = np.sqrt(((xx - cx) / (w * 0.75)) ** 2 + ((yy - cy) / (h * 0.75)) ** 2)
    r = np.clip(r, 0, 1)
    bg = (c_hi[None, None] * (1 - r[..., None]) + c_deep[None, None] * r[..., None])
    halo = np.exp(-r * 2.2)[..., None]
    bg = bg * 0.88 + 255.0 * 0.12 * halo

    # --- posterized holds (keep surface detail, lift a bit for sticker feel) ---
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    def quantize(x, levels):
        return np.round(x * (levels - 1) / 255.0) * (255.0 / (levels - 1))

    lab[..., 0] = np.clip(quantize(lab[..., 0], 6) + 14.0, 0, 255)  # L + lift
    lab[..., 1] = quantize(lab[..., 1], 8)   # more levels -> no hue smearing
    lab[..., 2] = quantize(lab[..., 2], 8)
    post = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)

    for hld in holds:
        bg[hld.mask] = post[hld.mask]

    out = bg.astype(np.uint8)

    # --- light ink edges (dark backdrop needs light lines) ---
    gray = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
    out[edges > 0] = np.clip(out[edges > 0].astype(int) * 1.35, 0, 255).astype(np.uint8)

    # --- matched route: lift + double outline (dark under-edge, white top) ---
    for hid in matched_ids:
        hld = next(x for x in holds if x.id == hid)
        ys, xs = np.nonzero(hld.mask)
        px = out[ys, xs].astype(np.float32)
        lab_px = cv2.cvtColor(px[None].astype(np.uint8), cv2.COLOR_RGB2LAB)[0].astype(np.float32)
        lab_px[..., 0] = np.clip(lab_px[..., 0] * 1.18, 0, 255)  # lift vs navy bg
        lab_px[..., 1] *= 1.25   # modest bump — heavy boost looked garish
        lab_px[..., 2] *= 1.25
        out[ys, xs] = cv2.cvtColor(lab_px[None].astype(np.uint8), cv2.COLOR_LAB2RGB)[0]
        outline(out, hld, (18, 26, 40), 6)    # dark under-edge (sticker shadow)
        outline(out, hld, (255, 255, 255), 3)  # white top edge


def route_order(holds: list[Hold], matched_ids: list[int]) -> list[int]:
    """Order matched holds by y (routes run bottom-up; y grows downward)."""
    return sorted(matched_ids, key=lambda hid: holds[hid].centroid[1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", type=Path,
                    default=Path("data/realpic/66ced02810271ec75cddbac23909f407.jpg"))
    ap.add_argument("--model", type=Path,
                    default=Path("output/onnx_640_aug/rfdetr-seg-small-aug.onnx"))
    ap.add_argument("--seed-lab", type=str, default="35 5 -25", help="target Lab 'L a b'")
    ap.add_argument("--out", type=Path, default=Path("data/route_viz"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lab = np.array([float(x) for x in args.seed_lab.split()])
    main(args.photo, args.model, lab, args.out)
