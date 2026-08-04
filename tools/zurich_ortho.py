#!/usr/bin/env python3
"""
Fetch SWISSIMAGE aerial photography covering the Zurich world and stitch it into
one ground texture.

This is what makes the ground look like the real place rather than like a colour:
real pavement, real tram beds, real courtyards, the river, the trees, the parks —
all of it photographed rather than invented.

Tiles are requested in swisstopo's **LV95 tile matrix set**, not Web Mercator.
The world's local frame is LV95 minus an origin, so LV95 tiles map onto it by
subtraction, with no reprojection and no resampling error.
"""

from __future__ import annotations

import io
import json
import pathlib
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

HERE = pathlib.Path(__file__).parent
CACHE = HERE / "cache" / "ortho"
CACHE.mkdir(parents=True, exist_ok=True)

LAYER = "ch.swisstopo.swissimage"
TILE = 256
ORIGIN_E, ORIGIN_N = 2420000.0, 1350000.0

# swisstopo's LV95 tile grid resolutions, metres per pixel, indexed by zoom level.
RESOLUTIONS = [4000, 3750, 3500, 3250, 3000, 2750, 2500, 2250, 2000, 1750,
               1500, 1250, 1000, 750, 650, 500, 250, 100, 50, 20, 10, 5,
               2.5, 2, 1.5, 1, 0.5, 0.25, 0.1]

LEVEL = int(sys.argv[1]) if len(sys.argv) > 1 else 26   # 26 = 0.5 m/px


def fetch(level: int, col: int, row: int) -> bytes | None:
    key = CACHE / f"{level}_{col}_{row}.jpg"
    if key.exists():
        return key.read_bytes()
    url = (f"https://wmts.geo.admin.ch/1.0.0/{LAYER}/default/current/2056/"
           f"{level}/{col}/{row}.jpeg")
    for _ in range(3):
        try:
            with urllib.request.urlopen(url, timeout=40) as resp:
                data = resp.read()
            key.write_bytes(data)
            return data
        except Exception:  # noqa: BLE001
            pass
    return None


def build() -> None:
    world = json.loads((HERE / "zurich_world.json").read_text())
    e0 = world["origin"]["east"]
    n0 = world["origin"]["north"]
    grid = world["terrain"]
    x0, z0 = grid["x0"], grid["z0"]
    x1 = x0 + (grid["nx"] - 1) * grid["cell"]
    z1 = z0 + (grid["nz"] - 1) * grid["cell"]

    res = RESOLUTIONS[LEVEL]
    span = res * TILE

    # World X is east, world Z is south. LV95 north decreases as Z increases.
    east_min, east_max = e0 + x0, e0 + x1
    north_max, north_min = n0 - z0, n0 - z1

    col0 = int((east_min - ORIGIN_E) // span)
    col1 = int((east_max - ORIGIN_E) // span)
    row0 = int((ORIGIN_N - north_max) // span)
    row1 = int((ORIGIN_N - north_min) // span)

    cols = col1 - col0 + 1
    rows = row1 - row0 + 1
    print(f"  level {LEVEL} = {res} m/px, {cols}x{rows} = {cols*rows} tiles")
    print(f"  output {cols*TILE}x{rows*TILE} px")

    jobs = [(c, r) for r in range(row0, row1 + 1) for c in range(col0, col1 + 1)]
    done = 0

    def grab(job):
        nonlocal done
        data = fetch(LEVEL, job[0], job[1])
        done += 1
        if done % 100 == 0:
            print(f"    {done}/{len(jobs)}", flush=True)
        return job, data

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(grab, jobs))

    canvas = Image.new("RGB", (cols * TILE, rows * TILE), (110, 115, 105))
    missing = 0
    for (c, r), data in results:
        if not data:
            missing += 1
            continue
        try:
            tile = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:  # noqa: BLE001
            missing += 1
            continue
        canvas.paste(tile, ((c - col0) * TILE, (r - row0) * TILE))

    out = HERE / "zurich_ortho.jpg"
    canvas.save(out, "JPEG", quality=88, optimize=True)

    # The exact world-space rectangle the texture covers, so the renderer can map
    # world XZ straight to UV without guessing.
    meta = {
        "level": LEVEL,
        "metresPerPixel": res,
        "width": cols * TILE,
        "height": rows * TILE,
        "worldMinX": (ORIGIN_E + col0 * span) - e0,
        "worldMinZ": n0 - (ORIGIN_N - row0 * span),
        "worldMaxX": (ORIGIN_E + (col1 + 1) * span) - e0,
        "worldMaxZ": n0 - (ORIGIN_N - (row1 + 1) * span),
        "attribution": "SWISSIMAGE (c) swisstopo",
    }
    (HERE / "zurich_ortho.json").write_text(json.dumps(meta, indent=2))

    print(f"  {missing} tiles missing")
    print(f"  wrote {out} ({out.stat().st_size/1e6:.1f} MB)")
    print(f"  covers X {meta['worldMinX']:.0f}..{meta['worldMaxX']:.0f}, "
          f"Z {meta['worldMinZ']:.0f}..{meta['worldMaxZ']:.0f}")


if __name__ == "__main__":
    build()
