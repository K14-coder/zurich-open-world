#!/usr/bin/env python3
"""
Replace estimated building heights with the measured ones.

swissBUILDINGS3D gives a full height — ground to roof apex — whereas our
extrusion is a wall that a procedural roof is then added on top of. So the wall
height to write is the measured total minus the roof rise the renderer will put
back, otherwise every building grows by the height of its own roof.

Matching is nearest centroid. The two datasets disagree slightly on what counts
as one building (12,277 measured against 12,538 in OSM: courtyards, annexes and
terraces get divided differently), so a match is only accepted when the nearest
measured building is both close and clearly closer than the runner-up.
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent / "data"

MAX_MATCH = 14.0     # metres between centroids
# The renderer's roof rise, mirrored from Roofs.swift: min(6.5, span * pitch),
# pitch 0.42..0.64. Using the midpoint is close enough — the error is a fraction
# of a metre against estimates that were wrong by ten.
ROOF_PITCH = 0.53


def roof_rise(ring) -> float:
    xs = [p[0] for p in ring]
    zs = [p[1] for p in ring]
    span = min(max(xs) - min(xs), max(zs) - min(zs))
    return min(6.5, max(1.4, span * ROOF_PITCH))


def build() -> None:
    measured = json.loads((DATA / "buildings3d.json").read_text())["buildings"]
    buildings = json.loads((DATA / "zurich_buildings.json").read_text())["buildings"]
    if not measured:
        raise SystemExit("no measured buildings; run swissbuildings.py first")

    pts = np.array([[b["x"], b["z"]] for b in measured])
    heights = np.array([b["zmax"] - b["zmin"] for b in measured])

    matched, ambiguous, far = 0, 0, 0
    deltas = []
    for b in buildings:
        ring = b["r"]
        cx = sum(p[0] for p in ring) / len(ring)
        cz = sum(p[1] for p in ring) / len(ring)
        d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cz)
        order = np.argsort(d)
        best = int(order[0])
        if d[best] > MAX_MATCH:
            far += 1
            continue
        # Two measured buildings equally close means the footprints are divided
        # differently here; taking either one's height is a coin flip.
        if len(order) > 1 and d[order[1]] < d[best] * 1.35 and d[order[1]] < 6.0:
            ambiguous += 1
            continue

        total = float(heights[best])
        if not (2.5 <= total <= 130):
            continue
        wall = max(2.5, total - roof_rise(ring))
        deltas.append(wall - b["h"])
        b["h"] = round(wall, 2)
        b["measured"] = True
        matched += 1

    (DATA / "zurich_buildings.json").write_text(
        json.dumps({"buildings": buildings}, separators=(",", ":")))

    a = np.array(deltas) if deltas else np.zeros(1)
    print(f"  {matched} of {len(buildings)} buildings now measured "
          f"({100*matched/len(buildings):.0f}%)")
    print(f"  {far} had no measured building within {MAX_MATCH:.0f} m, "
          f"{ambiguous} were ambiguous")
    print(f"  correction {a.mean():+.1f} m mean, {np.median(a):+.1f} m median")
    print(f"  {(a > 0).mean()*100:.0f}% were too short, "
          f"{(a < 0).mean()*100:.0f}% too tall")
    print(f"  worst overestimate {a.min():+.1f} m — that is the artifact")


if __name__ == "__main__":
    build()
