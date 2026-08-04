#!/usr/bin/env python3
"""
Turn OSM building footprints into extrudable prisms for the Zurich world.

Buildings are what make the city legible from the driver's seat — without them
a road network reads as abstract ribbons. OSM gives footprints for essentially
every building, but almost never a height: of 12,586 buildings here, 64 carry an
explicit height and 3,607 a storey count. The rest are estimated.

Emits footprint rings in the same local metric frame as the roads, each with a
base elevation sampled from the terrain grid, a height, and a triangulated roof.
"""

from __future__ import annotations

import json
import math
import pathlib

from zurich_circuit import wgs84_to_lv95
from zurich_world import terrain_at

HERE = pathlib.Path(__file__).parent
STOREY = 3.2  # metres per level


def estimate_height(tags: dict, area: float, seed: int) -> float:
    if "height" in tags:
        try:
            return max(2.5, min(120.0, float(str(tags["height"]).split()[0])))
        except ValueError:
            pass
    if "building:levels" in tags:
        try:
            levels = float(str(tags["building:levels"]).split(";")[0])
            return max(2.5, min(120.0, levels * STOREY + 1.2))
        except ValueError:
            pass

    kind = tags.get("building", "yes")
    if kind in ("garage", "garages", "shed", "hut", "carport", "roof", "kiosk"):
        return 3.0
    if kind in ("church", "cathedral"):
        return 26.0
    if kind in ("industrial", "warehouse", "retail", "commercial"):
        return 11.0
    if area < 45:
        return 3.5

    # Central Zurich is overwhelmingly four to six storeys. A single constant
    # produces a conspicuously flat, cardboard skyline, so vary it per building
    # off the way id — deterministic, so successive builds stay identical.
    base = 5.0 if area > 300 else 4.0
    jitter = ((seed * 2654435761) % 1000) / 1000.0  # Knuth multiplicative hash
    return (base + jitter * 2.0 - 0.5) * STOREY + 1.2


def signed_area(ring: list) -> float:
    s = 0.0
    for i in range(len(ring)):
        x1, z1 = ring[i]
        x2, z2 = ring[(i + 1) % len(ring)]
        s += x1 * z2 - x2 * z1
    return s / 2.0


def ear_clip(ring: list) -> list[int]:
    """Triangulate a simple polygon. Roofs are only visible from above, but a
    hole in every roof is glaring the moment the camera lifts at all."""
    n = len(ring)
    if n < 3:
        return []
    idx = list(range(n))
    if signed_area(ring) < 0:
        idx.reverse()

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def inside(p, a, b, c):
        d1 = cross(a, b, p); d2 = cross(b, c, p); d3 = cross(c, a, p)
        neg = d1 < 0 or d2 < 0 or d3 < 0
        pos = d1 > 0 or d2 > 0 or d3 > 0
        return not (neg and pos)

    tris = []
    guard = 0
    while len(idx) > 3 and guard < 5000:
        guard += 1
        clipped = False
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = ring[i0], ring[i1], ring[i2]
            if cross(a, b, c) <= 0:
                continue  # reflex
            if any(inside(ring[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue
            tris += [i0, i1, i2]
            idx.pop(k)
            clipped = True
            break
        if not clipped:
            break  # degenerate ring; keep what we have
    if len(idx) == 3:
        tris += idx
    return tris


def build() -> None:
    world = json.loads((HERE / "zurich_world.json").read_text())
    grid = world["terrain"]
    e0 = world["origin"]["east"]
    n0 = world["origin"]["north"]

    raw = json.loads((HERE / "buildings_raw.json").read_text())
    ways = [el for el in raw["elements"] if el.get("type") == "way"]

    out = []
    skipped = 0
    for way in ways:
        geom = way.get("geometry", [])
        if len(geom) < 4:
            skipped += 1
            continue
        ring = []
        for p in geom:
            east, north = wgs84_to_lv95(p["lat"], p["lon"])
            ring.append((east - e0, -(north - n0)))
        if ring[0] == ring[-1]:
            ring.pop()
        if len(ring) < 3:
            skipped += 1
            continue

        area = abs(signed_area(ring))
        if area < 8:
            skipped += 1
            continue

        cx = sum(p[0] for p in ring) / len(ring)
        cz = sum(p[1] for p in ring) / len(ring)
        base = terrain_at(grid, cx, cz)
        height = estimate_height(way.get("tags", {}), area, way["id"])
        roof = ear_clip(ring)
        if not roof:
            skipped += 1
            continue

        out.append({
            "r": [[round(x, 2), round(z, 2)] for x, z in ring],
            "b": round(base, 2),
            "h": round(height, 2),
            "t": roof,
        })

    path = HERE / "zurich_buildings.json"
    path.write_text(json.dumps({"buildings": out}, separators=(",", ":")))
    heights = [b["h"] for b in out]
    verts = sum(len(b["r"]) for b in out)
    tris = sum(len(b["t"]) // 3 + len(b["r"]) * 2 for b in out)
    print(f"  {len(out)} buildings ({skipped} skipped), {verts} footprint vertices")
    print(f"  heights {min(heights):.1f}..{max(heights):.1f} m, "
          f"mean {sum(heights)/len(heights):.1f} m")
    print(f"  ~{tris} triangles")
    print(f"  wrote {path} ({path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    build()
