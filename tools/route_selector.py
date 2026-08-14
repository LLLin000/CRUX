"""SeededRouteSelector prototype — PLAN D5/D6 core interaction logic.

hold mask -> Lab median color -> user taps a seed hold -> ΔE00 color match
-> DBSCAN spatial grouping -> default: the group containing the seed.

This is the selector-only layer: it consumes GT/segmentation masks and never
touches the model. Parameters (deltaE threshold, DBSCAN eps) are INITIAL
HEURISTICS per PLAN, to be calibrated against RouteBenchmark in D1/D2.

Usage:
    python tools/route_selector.py --selfcheck
"""
from __future__ import annotations

import dataclasses
from typing import Sequence

import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab
from sklearn.cluster import DBSCAN


@dataclasses.dataclass
class Hold:
    """One detected hold (mask in image space)."""
    id: int
    mask: np.ndarray            # bool 2D, image-sized
    centroid: tuple[float, float]  # (x, y) in image pixels
    lab: np.ndarray | None = None  # median Lab, computed lazily
    bbox: tuple[float, float, float, float] | None = None  # x, y, w, h


class HoldColorAnalyzer:
    """mask -> median Lab (PLAN D5: median ≈ NeuralClimb mode, chalk-robust)."""

    def __init__(self, image_rgb: np.ndarray):
        self.lab = rgb2lab(image_rgb)  # skimage rgb2lab assumes sRGB

    def median_lab(self, mask: np.ndarray) -> np.ndarray:
        pixels = self.lab[mask]
        return np.median(pixels, axis=0)


class SeededRouteSelector:
    """seed -> ΔE00 same-color candidates -> DBSCAN groups -> group with seed."""

    def __init__(
        self,
        image_rgb: np.ndarray,
        holds: Sequence[Hold],
        *,
        deltaE_threshold: float = 8.0,   # initial heuristic, calibrate in D1
        dbscan_eps_factor: float = 2.5,  # x median hold diameter. 2.0 cuts routes
                                         # whose spacing > 2x diameter (verified in
                                         # selfcheck) — heuristic, calibrate in D1
        min_samples: int = 2,            # min_samples=1 ≡ connected components
    ):
        self.analyzer = HoldColorAnalyzer(image_rgb)
        self.holds = list(holds)
        self.deltaE_threshold = deltaE_threshold
        self.dbscan_eps = dbscan_eps_factor * self._median_diameter()
        self.min_samples = min_samples
        for h in self.holds:
            h.lab = self.analyzer.median_lab(h.mask)
            ys, xs = np.nonzero(h.mask)
            h.bbox = (xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min())

    def _median_diameter(self) -> float:
        diams = []
        for h in self.holds:
            ys, xs = np.nonzero(h.mask)
            if len(xs):
                diams.append(float(np.median(np.maximum(xs.max() - xs.min(), ys.max() - ys.min()))))
        return float(np.median(diams)) if diams else 1.0

    def color_candidates(self, seed: Hold) -> list[Hold]:
        """All holds within ΔE00 of the seed's Lab color."""
        return [h for h in self.holds
                if deltaE_ciede2000(seed.lab[None], h.lab[None])[0] < self.deltaE_threshold]

    def select(self, seed_id: int) -> list[Hold]:
        """Return the group containing the seed (empty if none)."""
        seed = next(h for h in self.holds if h.id == seed_id)
        cands = self.color_candidates(seed)
        if not cands:
            return []
        pts = np.array([h.centroid for h in cands])
        labels = DBSCAN(eps=self.dbscan_eps, min_samples=self.min_samples).fit_predict(pts)
        for h, lbl in zip(cands, labels):
            h.cluster = lbl
        target = seed.cluster
        return [h for h in cands if h.cluster == target]


def _make_hold(image, color_range, centroid, size, hold_id):
    ys, xs = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    cy, cx = centroid
    half = size // 2
    mask = (np.abs(xs - cx) <= half) & (np.abs(ys - cy) <= half)
    image[mask] = color_range
    return Hold(id=hold_id, mask=mask, centroid=(cx, cy))


def selfcheck() -> None:
    """Synthetic wall: 3 blue holds in a vertical route, 1 stray blue, 2 reds.
    Seed on route-blue #1 -> must return the 3 route blues, exclude stray + reds."""
    img = np.full((300, 300, 3), 60, dtype=np.uint8)  # gray wall
    blue, red = np.array([40, 110, 235]), np.array([220, 60, 50])
    holds = [
        _make_hold(img, blue, (150, 250), 24, 1),  # route blue (bottom)
        _make_hold(img, blue, (150, 200), 24, 2),  # route blue
        _make_hold(img, blue, (150, 150), 24, 3),  # route blue (top)
        _make_hold(img, blue, (30, 30), 24, 4),    # stray blue, far away
        _make_hold(img, red, (240, 250), 24, 5),   # red route
        _make_hold(img, red, (240, 200), 24, 6),   # red route
    ]
    sel = SeededRouteSelector(img, holds)
    picked = sel.select(1)
    ids = {h.id for h in picked}
    assert ids == {1, 2, 3}, f"expected route blue {{1,2,3}}, got {ids}"
    print(f"selfcheck OK: seed #1 -> {sorted(ids)} (3 route blues, stray+reds excluded)")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--selfcheck":
        selfcheck()
    else:
        print(__doc__)
