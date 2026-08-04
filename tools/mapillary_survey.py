#!/usr/bin/env python3
"""
Survey Mapillary coverage before building anything on top of it.

The clean-plate approach lives or dies on how many separate passes exist down a
given street: the median can only outvote a parked car if the same façade was
photographed several times. Coverage is crowd-sourced and wildly uneven, so
picking the hero district by guesswork would mean discovering the gaps after
building the whole pipeline.

Counts, per façade, how many *distinct sequences* photographed it from an angle
that can actually see it — a sequence being one continuous capture run, which is
the right unit because two images from the same run share the same parked cars.

Token is read from ~/.config/mapillary/token, never from the repo.
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

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent / "data"
CACHE = HERE / "cache" / "mapillary"
CACHE.mkdir(parents=True, exist_ok=True)

BBOX = (47.3600, 8.5100, 47.3900, 8.5600)   # south, west, north, east
CELL = 0.004                                 # degrees per request tile
FIELDS = "id,computed_geometry,compass_angle,captured_at,is_pano,sequence"

# A photograph can only texture a wall it is both near enough to resolve and
# pointed at. 30 m and 60° off the wall normal are generous but not absurd.
MAX_RANGE = 30.0
MAX_OFF_AXIS = math.radians(65)


def token() -> str:
    path = pathlib.Path.home() / ".config" / "mapillary" / "token"
    if not path.exists():
        raise SystemExit("no token at ~/.config/mapillary/token")
    return path.read_text().strip()


def fetch_cell(args) -> list:
    south, west, tok = args
    north, east = south + CELL, west + CELL
    key = CACHE / f"{south:.4f}_{west:.4f}.json"
    if key.exists():
        return json.loads(key.read_text()).get("data", [])
    url = ("https://graph.mapillary.com/images?"
           + urllib.parse.urlencode({
               "access_token": tok,
               "bbox": f"{west},{south},{east},{north}",
               "fields": FIELDS,
               "limit": 2000,
           }))
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=90) as resp:
                payload = json.loads(resp.read())
            if "error" in payload:
                raise RuntimeError(payload["error"].get("message"))
            key.write_text(json.dumps(payload))
            return payload.get("data", [])
        except Exception as exc:  # noqa: BLE001
            if attempt == 3:
                print(f"  ! cell {south:.3f},{west:.3f}: {exc}", file=sys.stderr)
                return []
            time.sleep(1.5 * (attempt + 1))
    return []


def survey() -> list:
    tok = token()
    s, w, n, e = BBOX
    cells = []
    lat = s
    while lat < n:
        lon = w
        while lon < e:
            cells.append((lat, lon, tok))
            lon += CELL
        lat += CELL
    print(f"  querying {len(cells)} cells...")

    images = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, batch in enumerate(pool.map(fetch_cell, cells)):
            images.extend(batch)
            if (i + 1) % 25 == 0:
                print(f"    {i+1}/{len(cells)} cells, {len(images)} images", flush=True)

    # Cells overlap at their edges; the same image can come back twice.
    unique = {img["id"]: img for img in images if img.get("computed_geometry")}
    return list(unique.values())


def main() -> None:
    world = json.loads((DATA / "zurich_world.json").read_text())
    e0, n0 = world["origin"]["east"], world["origin"]["north"]
    facades = json.loads((DATA / "zurich_facades.json").read_text())["facades"]

    print("Surveying Mapillary coverage...")
    images = survey()
    if not images:
        raise SystemExit("no imagery returned")

    from zurich_circuit import wgs84_to_lv95

    panos = sum(1 for i in images if i.get("is_pano"))
    seqs = {i.get("sequence") for i in images if i.get("sequence")}
    years = {}
    for i in images:
        ts = i.get("captured_at")
        if ts:
            years[time.gmtime(ts / 1000).tm_year] = years.get(time.gmtime(ts / 1000).tm_year, 0) + 1
    print(f"\n  {len(images)} images, {len(seqs)} sequences, "
          f"{panos} panoramic ({100*panos/len(images):.0f}%)")
    print("  by year: " + ", ".join(f"{y}:{c}" for y, c in sorted(years.items())))

    # Project images into the world frame and grid them for lookup.
    GRID = 40.0
    index: dict[tuple[int, int], list] = {}
    for img in images:
        lon, lat = img["computed_geometry"]["coordinates"]
        east, north = wgs84_to_lv95(lat, lon)
        x, z = east - e0, -(north - n0)
        # Mapillary compass angle is degrees clockwise from north. World +X is
        # east and world +Z is south, so the viewing direction is (sin, cos)
        # with Z negated to point north at 0.
        a = math.radians(img.get("compass_angle") or 0.0)
        img["_p"] = (x, z, math.sin(a), -math.cos(a), img.get("sequence"))
        index.setdefault((int(x // GRID), int(z // GRID)), []).append(img)

    print("\n  matching images to façades...")
    reach = int(MAX_RANGE // GRID) + 1
    covered = 0
    passes_hist: dict[int, int] = {}
    per_street: dict[str, list] = {}

    for f in facades:
        mx = (f["a"][0] + f["z"][0]) / 2
        mz = (f["a"][1] + f["z"][1]) / 2
        nx, nz = f["n"]
        seen = set()
        ci, cj = int(mx // GRID), int(mz // GRID)
        for i in range(ci - reach, ci + reach + 1):
            for j in range(cj - reach, cj + reach + 1):
                for img in index.get((i, j), []):
                    x, z, dx, dz, seq = img["_p"]
                    vx, vz = mx - x, mz - z
                    dist = math.hypot(vx, vz)
                    if dist > MAX_RANGE or dist < 0.5:
                        continue
                    # Camera must be on the outward side of the wall...
                    if (x - mx) * nx + (z - mz) * nz <= 0:
                        continue
                    # ...and looking at it, not merely near it.
                    if (vx * dx + vz * dz) / dist < math.cos(MAX_OFF_AXIS):
                        continue
                    seen.add(seq)
        n_passes = len(seen)
        f["passes"] = n_passes
        passes_hist[min(n_passes, 10)] = passes_hist.get(min(n_passes, 10), 0) + 1
        if n_passes:
            covered += 1
        per_street.setdefault(f["road"] or "(unnamed)", []).append(n_passes)

    print(f"  {covered}/{len(facades)} façades seen by at least one pass "
          f"({100*covered/len(facades):.0f}%)")
    usable = sum(1 for f in facades if f["passes"] >= 4)
    print(f"  {usable} façades with >=4 passes ({100*usable/len(facades):.0f}%) "
          f"— the threshold the median needs")
    print("\n  passes per façade:")
    for k in sorted(passes_hist):
        label = f"{k}+" if k == 10 else str(k)
        print(f"    {label:>3}: {passes_hist[k]}")

    ranked = sorted(
        ((name, len(v), sum(1 for p in v if p >= 4)) for name, v in per_street.items()
         if name != "(unnamed)"),
        key=lambda r: -r[2])
    print("\n  best-covered streets (façades with >=4 passes):")
    for name, total, good in ranked[:12]:
        print(f"    {name[:34]:34s} {good:4d}/{total:4d}")

    (DATA / "zurich_facades.json").write_text(
        json.dumps({"facades": facades}, separators=(",", ":")))
    print(f"\n  wrote pass counts back into zurich_facades.json")


if __name__ == "__main__":
    main()
