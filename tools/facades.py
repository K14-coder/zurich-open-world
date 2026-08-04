#!/usr/bin/env python3
"""
Extract the façade planes that street-level imagery has to cover.

A building has walls on all sides, but only the ones facing a street were ever
photographed from one, and only those are ever seen from a car. Texturing the
rest is wasted capture, wasted atlas space and wasted compute — so the first
question in the whole pipeline is: which walls actually matter?

Emits one record per street-facing façade: its plan segment, outward normal,
base elevation, height, and the road it faces. That record is what the imagery
stage associates photographs with.
"""

from __future__ import annotations

import json
import math
import pathlib

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent / "data"

# How far a wall can sit from a road centreline and still be photographable from
# it. Beyond this the wall is in a courtyard or a back garden.
MAX_ROAD_DISTANCE = 30.0
# Walls shorter than this are chamfers and bay returns, not façades worth their
# own texture.
MIN_FACADE_LENGTH = 2.0
GRID = 40.0


def signed_area(ring: list) -> float:
    s = 0.0
    for i in range(len(ring)):
        x1, z1 = ring[i]
        x2, z2 = ring[(i + 1) % len(ring)]
        s += x1 * z2 - x2 * z1
    return s / 2.0


def point_in_ring(px: float, pz: float, ring: list) -> bool:
    inside = False
    n = len(ring)
    for i in range(n):
        x1, z1 = ring[i]
        x2, z2 = ring[(i + 1) % n]
        if (z1 > pz) != (z2 > pz):
            t = (pz - z1) / (z2 - z1)
            if px < x1 + t * (x2 - x1):
                inside = not inside
    return inside


def build_road_index(edges: list) -> dict:
    """Uniform grid of road segments so the proximity test is O(handful)."""
    index: dict[tuple[int, int], list] = {}
    for edge in edges:
        pts = [p for p in edge["p"] if len(p) == 3]
        for k in range(len(pts) - 1):
            a, b = pts[k], pts[k + 1]
            seg = (a[0], a[2], b[0], b[2], edge.get("n", ""), edge.get("c", ""))
            minx, maxx = sorted((a[0], b[0]))
            minz, maxz = sorted((a[2], b[2]))
            for cx in range(int(minx // GRID), int(maxx // GRID) + 1):
                for cz in range(int(minz // GRID), int(maxz // GRID) + 1):
                    index.setdefault((cx, cz), []).append(seg)
    return index


def nearest_road(index: dict, x: float, z: float, limit: float):
    best = None
    cell_r = int(limit // GRID) + 1
    cx, cz = int(x // GRID), int(z // GRID)
    for i in range(cx - cell_r, cx + cell_r + 1):
        for j in range(cz - cell_r, cz + cell_r + 1):
            for (ax, az, bx, bz, name, cls) in index.get((i, j), []):
                dx, dz = bx - ax, bz - az
                L2 = dx * dx + dz * dz
                t = 0.0 if L2 < 1e-9 else max(0.0, min(1.0, ((x - ax) * dx + (z - az) * dz) / L2))
                px, pz = ax + dx * t, az + dz * t
                d = math.hypot(x - px, z - pz)
                if best is None or d < best[0]:
                    best = (d, px, pz, name, cls)
    if best and best[0] <= limit:
        return best
    return None


def build() -> None:
    world = json.loads((DATA / "zurich_world.json").read_text())
    buildings = json.loads((DATA / "zurich_buildings.json").read_text())["buildings"]
    index = build_road_index(world["edges"])
    print(f"  indexed roads from {len(world['edges'])} edges")

    facades = []
    considered = 0

    for bid, b in enumerate(buildings):
        ring = [(p[0], p[1]) for p in b["r"]]
        if len(ring) < 3:
            continue
        ccw = signed_area(ring) > 0

        for i in range(len(ring)):
            ax, az = ring[i]
            bx, bz = ring[(i + 1) % len(ring)]
            dx, dz = bx - ax, bz - az
            length = math.hypot(dx, dz)
            if length < MIN_FACADE_LENGTH:
                continue
            considered += 1

            # Outward normal: interior lies to the left of a CCW edge, so the
            # outward side is the right. Verified against the ring rather than
            # trusted, because a handful of OSM footprints are wound the other
            # way and a flipped normal points the camera into the building.
            nx, nz = (dz / length, -dx / length) if ccw else (-dz / length, dx / length)
            mx, mz = (ax + bx) / 2, (az + bz) / 2
            if point_in_ring(mx + nx * 0.4, mz + nz * 0.4, ring):
                nx, nz = -nx, -nz

            hit = nearest_road(index, mx + nx * 1.5, mz + nz * 1.5, MAX_ROAD_DISTANCE)
            if hit is None:
                continue
            distance, rx, rz, road_name, road_class = hit

            # The road must lie on the outward side, not behind the wall.
            if (rx - mx) * nx + (rz - mz) * nz < 0:
                continue

            facades.append({
                "b": bid,
                "a": [round(ax, 2), round(az, 2)],
                "z": [round(bx, 2), round(bz, 2)],
                "n": [round(nx, 4), round(nz, 4)],
                "base": b["b"],
                "h": b["h"],
                "len": round(length, 2),
                "d": round(distance, 1),
                "road": road_name,
                "cls": road_class,
            })

    (DATA / "zurich_facades.json").write_text(
        json.dumps({"facades": facades}, separators=(",", ":")))

    area = sum(f["len"] * f["h"] for f in facades)
    named = len({f["road"] for f in facades if f["road"]})
    lengths = sorted(f["len"] for f in facades)
    print(f"  {len(facades)} street-facing façades of {considered} walls "
          f"({100*len(facades)/max(1,considered):.0f}%)")
    print(f"  {area/1000:.1f} thousand m² of façade, across {named} named streets")
    print(f"  median façade {lengths[len(lengths)//2]:.1f} m long")
    # At 512 px across a 10 m façade that is ~5 cm/px, about what street imagery
    # actually resolves at 15 m.
    print(f"  at 5 cm/px that is ~{area*400/1e6:.0f} megapixels of texture")


if __name__ == "__main__":
    build()
