#!/usr/bin/env python3
"""
Build an open-world road network for Zurich that ApexSim can drive on.

Not a circuit. This exports the whole street network of central Zurich as a
graph of road edges plus a terrain height grid, which a `RoadNetwork`
conforming to ApexSim's existing `GroundProvider` protocol can sample. The
vehicle simulation itself needs no changes at all: it only ever asks the world
`sampleGround(x:z:)`.

Sources
  * OpenStreetMap (ODbL) — road centrelines, widths, one-ways, speed limits.
  * swisstopo swissALTI3D via the geo.admin height API — terrain.

Output
  * zurich_world.json — origin, terrain grid, road edges with baked elevation.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from zurich_circuit import wgs84_to_lv95, haversine, height_at

HERE = pathlib.Path(__file__).parent
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)

BBOX = (47.3600, 8.5100, 47.3900, 8.5600)  # south, west, north, east

DRIVABLE = (
    "motorway|motorway_link|trunk|trunk_link|primary|primary_link"
    "|secondary|secondary_link|tertiary|tertiary_link"
    "|residential|unclassified|living_street|service"
)

# Parking aisles and driveways are tagged `service` and make up half of all ways
# in the city centre. They are not streets you drive *through*, and keeping them
# would double the data for no gain.
SKIP_SERVICE = {"parking_aisle", "driveway", "drive-through", "emergency_access"}

# Fallback widths, metres, when OSM gives neither a width nor a lane count.
DEFAULT_WIDTH = {
    "motorway": 14.0, "motorway_link": 7.0,
    "trunk": 12.0, "trunk_link": 7.0,
    "primary": 12.0, "primary_link": 6.5,
    "secondary": 10.5, "secondary_link": 6.5,
    "tertiary": 9.0, "tertiary_link": 6.0,
    "residential": 7.5, "unclassified": 7.0,
    "living_street": 6.0, "service": 4.5,
}

TERRAIN_CELL = 80.0  # metres between terrain samples


# --------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------

def fetch_roads() -> list[dict]:
    raw = HERE / "roads_raw.json"
    if not raw.exists():
        s, w, n, e = BBOX
        query = (
            f"[out:json][timeout:180];"
            f'way["highway"~"^({DRIVABLE})$"]({s},{w},{n},{e});'
            f"out body geom;"
        )
        req = urllib.request.Request(
            "https://overpass.osm.ch/api/interpreter",
            data=urllib.parse.urlencode({"data": query}).encode(),
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            raw.write_bytes(resp.read())
    data = json.loads(raw.read_text())
    return [el for el in data["elements"] if el.get("type") == "way"]


def keep(way: dict) -> bool:
    tags = way.get("tags", {})
    if tags.get("highway") == "service" and tags.get("service") in SKIP_SERVICE:
        return False
    if tags.get("access") in ("private", "no"):
        return False
    if len(way.get("geometry", [])) < 2:
        return False
    return True


def width_of(tags: dict) -> float:
    if "width" in tags:
        try:
            return max(3.0, min(25.0, float(str(tags["width"]).split()[0])))
        except ValueError:
            pass
    if "lanes" in tags:
        try:
            return max(3.0, min(25.0, int(tags["lanes"]) * 3.25))
        except ValueError:
            pass
    return DEFAULT_WIDTH.get(tags.get("highway"), 6.0)


def speed_of(tags: dict) -> int:
    raw = tags.get("maxspeed", "")
    try:
        return int(str(raw).split()[0])
    except (ValueError, IndexError):
        hw = tags.get("highway", "")
        if hw.startswith("motorway"):
            return 100
        if hw.startswith(("trunk", "primary")):
            return 60
        if hw in ("living_street", "service"):
            return 20
        return 50


# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------

def split_at_junctions(ways: list[dict]) -> list[dict]:
    """Cut every way wherever another way touches it. The pieces between
    junctions are the edges of the drivable graph."""
    shared: dict[int, int] = {}
    for way in ways:
        for nid in way.get("nodes", []):
            shared[nid] = shared.get(nid, 0) + 1

    edges = []
    for way in ways:
        nodes, geom = way["nodes"], way["geometry"]
        tags = way.get("tags", {})
        cut = [0]
        for i in range(1, len(nodes) - 1):
            if shared.get(nodes[i], 0) >= 2:
                cut.append(i)
        cut.append(len(nodes) - 1)

        for a, b in zip(cut, cut[1:]):
            if b - a < 1:
                continue
            edges.append({
                "nodes": nodes[a:b + 1],
                "geom": [(p["lat"], p["lon"]) for p in geom[a:b + 1]],
                "name": tags.get("name", ""),
                "highway": tags.get("highway", ""),
                "width": width_of(tags),
                "speed": speed_of(tags),
                "oneway": tags.get("oneway") in ("yes", "true", "1"),
                "bridge": "bridge" in tags,
                "tunnel": "tunnel" in tags,
            })
    return edges


# --------------------------------------------------------------------------
# Terrain
# --------------------------------------------------------------------------

def build_terrain(e0: float, n0: float, half_x: float, half_z: float) -> dict:
    """A regular grid of swissALTI3D samples covering the world, so any XZ can be
    bilinearly interpolated. Sampling a grid rather than every road vertex is
    both far fewer requests and strictly better: roads and terrain then agree by
    construction, instead of roads floating above or sinking into the ground."""
    cache = CACHE / "terrain.json"
    if cache.exists():
        return json.loads(cache.read_text())

    nx = int(2 * half_x / TERRAIN_CELL) + 1
    nz = int(2 * half_z / TERRAIN_CELL) + 1
    x0, z0 = -half_x, -half_z
    coords = [
        (x0 + i * TERRAIN_CELL, z0 + j * TERRAIN_CELL)
        for j in range(nz) for i in range(nx)
    ]
    print(f"  sampling {len(coords)} terrain points ({nx}x{nz} at {TERRAIN_CELL:.0f} m)...")

    def sample(c):
        x, z = c
        return height_at(e0 + x, n0 - z)  # world Z is south, LV95 north is +N

    with ThreadPoolExecutor(max_workers=12) as pool:
        heights = list(pool.map(sample, coords))

    good = [h for h in heights if not math.isnan(h)]
    if not good:
        raise SystemExit("terrain sampling failed entirely")
    fill = sum(good) / len(good)
    heights = [fill if math.isnan(h) else h for h in heights]

    grid = {"x0": x0, "z0": z0, "cell": TERRAIN_CELL, "nx": nx, "nz": nz,
            "heights": [round(h, 2) for h in heights]}
    cache.write_text(json.dumps(grid))
    print(f"  terrain {min(heights):.1f}..{max(heights):.1f} m")
    return grid


def terrain_at(grid: dict, x: float, z: float) -> float:
    """Bilinear sample, clamped at the edges."""
    fx = (x - grid["x0"]) / grid["cell"]
    fz = (z - grid["z0"]) / grid["cell"]
    i = max(0, min(grid["nx"] - 2, int(fx)))
    j = max(0, min(grid["nz"] - 2, int(fz)))
    tx, tz = max(0.0, min(1.0, fx - i)), max(0.0, min(1.0, fz - j))
    h = grid["heights"]
    nx = grid["nx"]
    h00, h10 = h[j * nx + i], h[j * nx + i + 1]
    h01, h11 = h[(j + 1) * nx + i], h[(j + 1) * nx + i + 1]
    return (h00 * (1 - tx) * (1 - tz) + h10 * tx * (1 - tz)
            + h01 * (1 - tx) * tz + h11 * tx * tz)


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

def build() -> None:
    print("Fetching road network...")
    ways = [w for w in fetch_roads() if keep(w)]
    print(f"  {len(ways)} drivable ways")

    edges = split_at_junctions(ways)
    print(f"  split into {len(edges)} edges at junctions")

    # Local metric frame, origin at the centre of the box. X east, Z south, Y up.
    s, w, n, e = BBOX
    e0, n0 = wgs84_to_lv95((s + n) / 2, (w + e) / 2)
    ec, nc = wgs84_to_lv95(s, w)
    half_x = abs(e0 - ec)
    half_z = abs(n0 - nc)
    print(f"  world extent {2*half_x:.0f} x {2*half_z:.0f} m")

    print("Sampling swissALTI3D terrain...")
    grid = build_terrain(e0, n0, half_x, half_z)

    datum = min(grid["heights"])
    grid["heights"] = [round(h - datum, 2) for h in grid["heights"]]

    print("Baking road elevation...")
    out_edges = []
    bridges = 0
    for edge in edges:
        pts = []
        for lat, lon in edge["geom"]:
            east, north = wgs84_to_lv95(lat, lon)
            x, z = east - e0, -(north - n0)
            pts.append([x, terrain_at(grid, x, z), z])

        # A bridge deck does not follow the ground: under the Limmat bridges the
        # terrain model returns the water surface, which would dip the road into
        # the river. Run the deck straight from one abutment to the other.
        if edge["bridge"] and len(pts) >= 2:
            bridges += 1
            y0, y1 = pts[0][1], pts[-1][1]
            total = sum(math.dist(pts[i][::2], pts[i + 1][::2])
                        for i in range(len(pts) - 1)) or 1.0
            run = 0.0
            for i in range(1, len(pts) - 1):
                run += math.dist(pts[i - 1][::2], pts[i][::2])
                pts[i][1] = y0 + (y1 - y0) * (run / total)

        out_edges.append({
            "p": [[round(v, 2) for v in p] for p in pts],
            "w": round(edge["width"], 1),
            "n": edge["name"],
            "c": edge["highway"],
            "s": edge["speed"],
            "o": edge["oneway"],
            "b": edge["bridge"],
            "t": edge["tunnel"],
        })

    total_km = sum(
        sum(math.dist(e["p"][i][::2], e["p"][i + 1][::2]) for i in range(len(e["p"]) - 1))
        for e in out_edges
    ) / 1000.0

    world = {
        "name": "Zurich",
        "attribution": "Roads (c) OpenStreetMap contributors, ODbL. "
                       "Terrain (c) swisstopo swissALTI3D.",
        "origin": {"east": round(e0, 2), "north": round(n0, 2), "datum": round(datum, 2)},
        "bbox": BBOX,
        "terrain": grid,
        "edges": out_edges,
    }
    out = HERE / "zurich_world.json"
    out.write_text(json.dumps(world, separators=(",", ":")))

    named = len({e["n"] for e in out_edges if e["n"]})
    print(f"\n  {len(out_edges)} edges, {total_km:.1f} km of road, "
          f"{named} named streets, {bridges} bridges")
    print(f"  terrain relief {max(grid['heights']):.1f} m over the world")
    print(f"  wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    build()
