#!/usr/bin/env python3
"""
Measure building heights from the panoramas instead of guessing them.

Of 12,586 OSM buildings here, 64 carry an explicit height and 3,607 a storey
count; the rest were estimated from footprint area. Those guesses are the root
cause of the worst projection artifact: where our box stands taller than the
building does, the projection paints whatever the photograph holds at that
bearing — a tree, a distant roof — onto the surplus, and it reads as a slab of
scenery floating above the roofline.

No shader can fix a box that is five metres too tall. But the photograph knows
where the roof is: march a point up the façade, project it into the panorama at
each step, and the height where the segmentation turns to sky is the roofline.

Only touches buildings the panoramas can actually see. Everything else keeps its
estimate.
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent / "data"
PANOS = DATA / "panoramas"

MAX_RANGE = 90.0        # beyond this a roofline is a few pixels and not worth trusting
MIN_H, MAX_H = 3.0, 60.0
STEP = 0.5
SKY = 40                # mask value below this is sky (sky is written as 0)


def load_panoramas():
    index = json.loads((DATA / "panoramas.json").read_text())
    out = []
    for p in index["panoramas"]:
        mask_path = PANOS / (pathlib.Path(p["file"]).stem + "_mask.png")
        if not mask_path.exists():
            continue
        out.append({
            "pos": np.array(p["pos"], dtype=np.float64),
            "R": np.array(p["R"], dtype=np.float64).reshape(3, 3),
            "mask": np.asarray(Image.open(mask_path)),
        })
    return out


def sample_mask(pano, world: np.ndarray) -> int:
    """Segmentation class the panorama holds in the direction of `world`."""
    d = world - pano["pos"]
    # World is X east, Y up, Z south; the pose is against local ENU.
    enu = np.array([d[0], -d[2], d[1]])
    cam = pano["R"] @ enu
    lon = math.atan2(cam[0], cam[2])
    lat = math.atan2(-cam[1], math.hypot(cam[0], cam[2]))
    h, w = pano["mask"].shape
    px = int((0.5 + lon / (2 * math.pi)) * w) % w
    py = int((0.5 - lat / math.pi) * h)
    py = max(0, min(h - 1, py))
    return int(pano["mask"][py, px])


def roofline(pano, x: float, z: float, base: float) -> float | None:
    """Height at which this façade point stops being solid and becomes sky."""
    last_solid = None
    y = base + MIN_H
    while y <= base + MAX_H:
        if sample_mask(pano, np.array([x, y, z])) <= SKY:
            # Sky. If we never saw anything solid below it the ray missed the
            # building entirely — a gap between blocks, say — so report nothing.
            return None if last_solid is None else last_solid - base
        last_solid = y
        y += STEP
    return None      # still solid at 60 m: taller than we search, leave it alone


def build() -> None:
    panos = load_panoramas()
    if not panos:
        raise SystemExit("no panorama masks; run pano_masks.py first")
    buildings = json.loads((DATA / "zurich_buildings.json").read_text())["buildings"]
    centres = np.array([p["pos"] for p in panos])

    fixed, skipped, deltas = 0, 0, []
    for b in buildings:
        ring = b["r"]
        cx = sum(p[0] for p in ring) / len(ring)
        cz = sum(p[1] for p in ring) / len(ring)
        d = np.hypot(centres[:, 0] - cx, centres[:, 2] - cz)
        if d.min() > MAX_RANGE:
            skipped += 1
            continue
        pano = panos[int(np.argmin(d))]

        # Sample along the footprint, nudged outwards so the ray hits the façade
        # rather than starting inside the wall.
        found = []
        n = len(ring)
        for i in range(n):
            ax, az = ring[i]
            bx, bz = ring[(i + 1) % n]
            mx, mz = (ax + bx) / 2, (az + bz) / 2
            ex, ez = bx - ax, bz - az
            L = math.hypot(ex, ez)
            if L < 1.5:
                continue
            nx, nz = -ez / L, ex / L
            for sign in (1, -1):
                h = roofline(pano, mx + nx * 0.5 * sign, mz + nz * 0.5 * sign, b["b"])
                if h is not None and MIN_H <= h <= MAX_H:
                    found.append(h)

        # A median over several façade points, and only with enough agreement:
        # one stray reading is a lamp post or a passing van, not a roofline.
        if len(found) >= 4:
            measured = float(np.median(found))
            if abs(measured - b["h"]) > 1.0:
                deltas.append(measured - b["h"])
                b["h"] = round(measured, 2)
                fixed += 1
        else:
            skipped += 1

    (DATA / "zurich_buildings.json").write_text(
        json.dumps({"buildings": buildings}, separators=(",", ":")))

    print(f"  {fixed} heights measured from the photographs, {skipped} left as estimated")
    if deltas:
        a = np.array(deltas)
        print(f"  correction {a.mean():+.1f} m mean, {np.median(a):+.1f} m median, "
              f"range {a.min():+.1f} to {a.max():+.1f}")
        print(f"  {(a < 0).mean()*100:.0f}% were too tall — the direction the "
              f"artifact predicted")


if __name__ == "__main__":
    build()
