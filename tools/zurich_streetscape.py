#!/usr/bin/env python3
"""
Fetch the things that actually make a street look like a street.

Façade detail was never the main gap. The streets were *empty* — no trees, no
tram rails, no kerbs, no lamps. A real street is dense with objects, and a
Zurich one is defined by plane trees and tram wire. OSM has all of it.

Emits a streetscape file in the same local metric frame as the roads, with
elevation baked from the terrain grid.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
import urllib.parse
import urllib.request

from zurich_circuit import wgs84_to_lv95
from zurich_world import terrain_at

HERE = pathlib.Path(__file__).parent
BBOX = (47.3600, 8.5100, 47.3900, 8.5600)
MIRRORS = ["https://overpass.osm.ch/api/interpreter",
           "https://overpass-api.de/api/interpreter"]


def overpass(query: str, cache_name: str) -> dict:
    cache = HERE / f"{cache_name}_raw.json"
    if cache.exists():
        return json.loads(cache.read_text())
    last = None
    for endpoint in MIRRORS:
        try:
            req = urllib.request.Request(
                endpoint, data=urllib.parse.urlencode({"data": query}).encode())
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = resp.read()
            cache.write_bytes(data)
            return json.loads(data)
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"  ! {endpoint}: {exc}", file=sys.stderr)
    raise SystemExit(f"every mirror failed: {last}")


def build() -> None:
    world = json.loads((HERE.parent / "data" / "zurich_world.json").read_text())
    grid = world["terrain"]
    e0, n0 = world["origin"]["east"], world["origin"]["north"]

    s, w, n, e = BBOX
    query = (
        f"[out:json][timeout:300];("
        f'node["natural"="tree"]({s},{w},{n},{e});'
        f'way["railway"="tram"]({s},{w},{n},{e});'
        f'node["highway"="street_lamp"]({s},{w},{n},{e});'
        f'node["highway"="crossing"]({s},{w},{n},{e});'
        f');out body geom;'
    )
    print("Fetching streetscape from OpenStreetMap...")
    data = overpass(query, "streetscape")

    def to_local(lat, lon):
        east, north = wgs84_to_lv95(lat, lon)
        return east - e0, -(north - n0)

    trees, lamps, crossings, rails = [], [], [], []

    for el in data.get("elements", []):
        tags = el.get("tags", {})
        if el["type"] == "node":
            x, z = to_local(el["lat"], el["lon"])
            y = terrain_at(grid, x, z)
            if tags.get("natural") == "tree":
                # Height is rarely tagged. Vary deterministically off the id so
                # a street of trees is not a row of identical clones.
                seed = (el["id"] * 2654435761) % 10007 / 10007.0
                height = 9.0 + seed * 8.0
                try:
                    if "height" in tags:
                        height = max(3.0, min(30.0, float(str(tags["height"]).split()[0])))
                except ValueError:
                    pass
                trees.append({"p": [round(x, 2), round(y, 2), round(z, 2)],
                              "h": round(height, 2), "s": round(seed, 4)})
            elif tags.get("highway") == "street_lamp":
                lamps.append({"p": [round(x, 2), round(y, 2), round(z, 2)]})
            elif tags.get("highway") == "crossing":
                crossings.append({"p": [round(x, 2), round(y, 2), round(z, 2)]})
        elif el["type"] == "way" and tags.get("railway") == "tram":
            pts = []
            for p in el.get("geometry", []):
                x, z = to_local(p["lat"], p["lon"])
                pts.append([round(x, 2), round(terrain_at(grid, x, z), 2), round(z, 2)])
            if len(pts) >= 2:
                rails.append({"p": pts})

    out = {"trees": trees, "lamps": lamps, "crossings": crossings, "rails": rails,
           "attribution": "(c) OpenStreetMap contributors, ODbL"}
    path = HERE.parent / "data" / "zurich_streetscape.json"
    path.write_text(json.dumps(out, separators=(",", ":")))

    railLen = sum(
        sum(math.dist(r["p"][i][::2], r["p"][i + 1][::2]) for i in range(len(r["p"]) - 1))
        for r in rails) / 1000.0
    print(f"  {len(trees)} trees, {len(lamps)} street lamps, "
          f"{len(crossings)} crossings")
    print(f"  {len(rails)} tram segments, {railLen:.1f} km of rail")
    print(f"  wrote {path} ({path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    build()
