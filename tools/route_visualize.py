"""Visual experiments: color-match route rendering on real photo annotations.

Pipeline: ONNX model detections -> per-hold Lab median -> seed color (blue)
-> dE00 color match -> DBSCAN grouping -> three renderings:
  1. highlight: matched holds popped, everything else dimmed
  2. badge: cartoonish flat-color route (chunky outline, posterized bg)
  3. route line: connects matched holds (group order) + cluster stats

Usage:
    python tools/route_visualize.py --photo data/realpic/66ced02810271ec75cddbac23909f407.jpg \
        --model output/onnx_v101/crux-hold-seg-v1.0.1-648-fp16.onnx --seed-lab "35 5 -25" --out data/route_viz
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

    # ---------- rendering 4: circular badge logo (holds extracted) ----------
    logo = badge_logo(img_rgb, holds, matched_ids, label=args.label)
    cv2.imwrite(str(out_dir / "4_badge_logo.png"), cv2.cvtColor(logo, cv2.COLOR_RGB2BGR))

    print(f"wrote {out_dir}/1_highlight.jpg, 2_badge.jpg, 3_route_line.jpg, 4_badge_logo.png")


def outline(img: np.ndarray, hold: Hold, color, thickness: int) -> None:
    """Smooth outline via the shared edge-preserving two-stage RDP polygon
    (same path as pre-annotations) — never the raw raster mask edge.
    reject_straight=False: long bar holds are real holds, keep polyline style.
    Fallback (degenerate mask) also approximates instead of pixel-drawing."""
    poly = mask_to_polygon(hold.mask, reject_straight=False)
    if poly is None:
        m = (hold.mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
        best = max(contours, key=cv2.contourArea)
        pts = cv2.approxPolyDP(best, 2.0, True).reshape(-1, 2)
        cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)
        return
    pts = np.array(poly[0]).reshape(-1, 2).astype(np.int32)
    cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)


def cartoon(img_rgb: np.ndarray, holds: list[Hold], matched_ids: list[int]) -> np.ndarray:
    """Badge-style cartoon that NEVER recolors holds:
    - backdrop: fixed navy radial gradient + soft center glow
    - ALL holds stamped from the ORIGINAL photo (true colors, full detail)
    - matched route: double outline (dark under-edge + white top edge) only
    - edges: light ink lines so the navy backdrop keeps some texture"""
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

    # --- holds: original pixels, unchanged color ---
    for hld in holds:
        bg[hld.mask] = img_rgb[hld.mask]

    out = bg.astype(np.uint8)

    # --- light ink edges, BACKGROUND ONLY (hold pixels stay untouched) ---
    gray = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
    all_holds = np.zeros((h, w), bool)
    for hld in holds:
        all_holds |= hld.mask
    ink = (edges & ~all_holds).astype(bool)  # bool mask, NOT 0/1 int indexing
    out[ink] = np.clip(out[ink].astype(int) * 1.35, 0, 255).astype(np.uint8)

    # --- matched route: outline only, colors untouched ---
    for hid in matched_ids:
        hld = next(x for x in holds if x.id == hid)
        outline(out, hld, (18, 26, 40), 6)    # dark under-edge (sticker shadow)
        outline(out, hld, (255, 255, 255), 3)  # white top edge
    return out


def badge_logo(img_rgb: np.ndarray, holds: list[Hold], matched_ids: list[int],
               size: int = 1200, label: str = "CRUX") -> np.ndarray:
    """Extract matched holds into a standalone circular badge logo.

    - navy radial-gradient disc + white ring (classic mark look)
    - holds keep their REAL colors, warped into the disc preserving the
      route's aspect ratio (no per-hold distortion), white polyline outlines
    - optional label centered under the route
    """
    mh = [next(h for h in holds if h.id == hid) for hid in matched_ids]
    if not mh:
        return np.full((size, size, 3), 20, np.uint8)

    # route bbox in photo space
    xs0, ys0 = np.inf, np.inf
    xs1, ys1 = -np.inf, -np.inf
    for h in mh:
        ys, xs = np.nonzero(h.mask)
        xs0, ys0 = min(xs0, xs.min()), min(ys0, ys.min())
        xs1, ys1 = max(xs1, xs.max()), max(ys1, ys.max())
    rw, rh = xs1 - xs0, ys1 - ys0
    pad = 0.08 * max(rw, rh)
    rw, rh = rw + 2 * pad, rh + 2 * pad

    # fit into disc interior (keep aspect ratio), slight upward bias for label
    inner = size * 0.78
    scale = min(inner / rw, inner / rh)
    dw, dh = rw * scale, rh * scale
    ox = (size - dw) / 2
    oy = (size - dh) / 2 - size * 0.04

    # --- disc backdrop ---
    yy, xx = np.mgrid[0:size, 0:size]
    c = np.sqrt(((xx - size / 2) / (size / 2)) ** 2 + ((yy - size / 2) / (size / 2)) ** 2)
    grad = (np.array([56, 74, 102])[None, None] * (1 - c[..., None]) +
            np.array([14, 20, 33])[None, None] * c[..., None])
    disc = grad.astype(np.uint8)
    # white ring
    ring = (c > 0.955) & (c < 0.995)
    disc[ring] = (240, 240, 240)
    # feather the disc edge
    disc[c > 0.995] = 0  # outside -> black (transparency handled at crop)

    # --- route line connecting holds (bottom-up order, drawn first) ---
    order = sorted(mh, key=lambda h: h.centroid[1])
    line_pts = []
    for h in order:
        ys, xs = np.nonzero(h.mask)
        cx = ox + (xs.mean() - xs0) * scale
        cy = oy + (ys.mean() - ys0) * scale
        line_pts.append((int(cx), int(cy)))
    for a, b in zip(line_pts[:-1], line_pts[1:]):
        cv2.line(disc, a, b, (230, 230, 235), max(2, size // 400), cv2.LINE_AA)

    # --- warp each hold mask+color into disc coords ---
    for h in mh:
        ys, xs = np.nonzero(h.mask)
        # map photo (x, y) -> disc (X, Y)
        X = ox + (xs.astype(np.float64) - xs0) * scale
        Y = oy + (ys.astype(np.float64) - ys0) * scale
        Xi, Yi = X.astype(np.int32), Y.astype(np.int32)
        keep = (Xi >= 0) & (Xi < size) & (Yi >= 0) & (Yi < size)
        Xi, Yi = Xi[keep], Yi[keep]
        col = img_rgb[ys[keep], xs[keep]]
        disc[Yi, Xi] = col  # nearest-neighbor warp (fine at this scale)
        outline_at(disc, Xi, Yi, (255, 255, 255), 3)

    # --- label ---
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = size / 16
        thick = max(3, size // 150)
        (tw, th), base = cv2.getTextSize(label, font, fs, thick)
        tx = (size - tw) // 2
        ty = size - int(size * 0.075)
        # subtle shadow for legibility on the dark disc
        cv2.putText(disc, label, (tx + 2, ty + 2), font, fs, (10, 15, 25), thick, cv2.LINE_AA)
        cv2.putText(disc, label, (tx, ty), font, fs, (240, 240, 240), thick, cv2.LINE_AA)

    return disc


def outline_at(img: np.ndarray, xs: np.ndarray, ys: np.ndarray, color, thickness: int) -> None:
    """White outline for a warped hold: paint pixels near the boundary."""
    m = np.zeros(img.shape[:2], np.uint8)
    m[ys, xs] = 255
    edge = m - cv2.erode(m, np.ones((2 * thickness + 1, 2 * thickness + 1), np.uint8))
    img[edge > 0] = color


def route_order(holds: list[Hold], matched_ids: list[int]) -> list[int]:
    """Order matched holds by y (routes run bottom-up; y grows downward)."""
    return sorted(matched_ids, key=lambda hid: holds[hid].centroid[1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", type=Path,
                    default=Path("data/realpic/66ced02810271ec75cddbac23909f407.jpg"))
    ap.add_argument("--model", type=Path,
                    default=Path("output/onnx_v101/crux-hold-seg-v1.0.1-648-fp16.onnx"))
    ap.add_argument("--seed-lab", type=str, default="35 5 -25", help="target Lab 'L a b'")
    ap.add_argument("--out", type=Path, default=Path("data/route_viz"))
    ap.add_argument("--label", type=str, default="CRUX", help="badge logo text ('' to skip)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lab = np.array([float(x) for x in args.seed_lab.split()])
    main(args.photo, args.model, lab, args.out)
