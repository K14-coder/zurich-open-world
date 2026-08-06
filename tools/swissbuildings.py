#!/usr/bin/env python3
"""
Real building heights and roof shapes from swisstopo swissBUILDINGS3D 3.0.

Of 12,586 OSM buildings in this world, 64 carry an explicit height and 3,607 a
storey count; the rest were estimated from footprint area. Those estimates are
the root cause of the worst projection artifact — where our box stands taller
than the building does, the projection paints whatever the photograph holds at
that bearing onto the surplus, and it reads as scenery floating above a roofline.

swissBUILDINGS3D is LoD2: every building has measured wall and roof surfaces in
LV95, with real elevations. It settles heights, and carries the actual roof forms
the procedural pitched roofs are standing in for.

The tiles are large — 88 MB zipped, 860 MB of XML each, four of them. Nothing is
ever extracted to disk: each zip member is streamed, split on building
boundaries, and a cheap coordinate test rejects the vast majority before any XML
parsing happens. Only this world's 12.6 km² is kept.
"""

from __future__ import annotations

import json
import math
import pathlib
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent / "data"
CACHE = HERE / "cache" / "sb3d"
CACHE.mkdir(parents=True, exist_ok=True)

STAC = ("https://data.geo.admin.ch/api/stac/v0.9/collections/"
        "ch.swisstopo.swissbuildings3d_3_0/items")

NS = {
    "bldg": "http://www.opengis.net/citygml/building/2.0",
    "gml": "http://www.opengis.net/gml",
}

MEMBER_START = "<core:cityObjectMember>"
MEMBER_END = "</core:cityObjectMember>"
# First coordinate pair of any posList, for the pre-filter.
FIRST_COORD = re.compile(r"<gml:posList[^>]*>\s*(\d{6,7}\.?\d*)\s+(\d{6,7}\.?\d*)")


def tile_urls(bbox) -> list[str]:
    s, w, n, e = bbox
    url = f"{STAC}?bbox={w},{s},{e},{n}&limit=50"
    with urllib.request.urlopen(url, timeout=120) as r:
        items = json.loads(r.read()).get("features", [])
    urls = []
    for it in items:
        for name, asset in (it.get("assets") or {}).items():
            if "citygml" in name:
                urls.append(asset["href"])
    return sorted(set(urls))


def fetch(url: str) -> pathlib.Path:
    path = CACHE / pathlib.Path(url).name
    if path.exists():
        return path
    print(f"    downloading {path.name} ...", flush=True)
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=1800) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(path)
    return path


def members(path: pathlib.Path):
    """Yield the XML text of each cityObjectMember, streaming from inside the zip."""
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".gml"))
        with z.open(name) as f:
            buffer = ""
            while chunk := f.read(1 << 22):
                buffer += chunk.decode("utf-8", "replace")
                while True:
                    start = buffer.find(MEMBER_START)
                    if start < 0:
                        # Keep only a tail that might hold a split start tag.
                        buffer = buffer[-len(MEMBER_START):]
                        break
                    end = buffer.find(MEMBER_END, start)
                    if end < 0:
                        buffer = buffer[start:]
                        break
                    yield buffer[start:end + len(MEMBER_END)]
                    buffer = buffer[end + len(MEMBER_END):]


def parse_member(xml_text: str):
    """Wall and roof surfaces of one building, as lists of LV95 triples."""
    # The member references namespaces declared on the document root, so give it
    # a wrapper carrying them rather than re-parsing the whole file.
    wrapper = (
        '<core:cityObjectMember xmlns:core="http://www.opengis.net/citygml/2.0" '
        'xmlns:bldg="http://www.opengis.net/citygml/building/2.0" '
        'xmlns:gml="http://www.opengis.net/gml" '
        'xmlns:gen="http://www.opengis.net/citygml/generics/2.0">'
        + xml_text[len(MEMBER_START):]
    )
    try:
        root = ET.fromstring(wrapper)
    except ET.ParseError:
        return None

    surfaces = {"roof": [], "wall": [], "ground": []}
    for kind, tag in (("roof", "RoofSurface"), ("wall", "WallSurface"),
                      ("ground", "GroundSurface")):
        for surf in root.iter(f"{{{NS['bldg']}}}{tag}"):
            for pos in surf.iter(f"{{{NS['gml']}}}posList"):
                nums = pos.text.split() if pos.text else []
                if len(nums) < 9 or len(nums) % 3:
                    continue
                ring = [(float(nums[i]), float(nums[i + 1]), float(nums[i + 2]))
                        for i in range(0, len(nums), 3)]
                surfaces[kind].append(ring)
    if not any(surfaces.values()):
        return None
    return surfaces


def build() -> None:
    world = json.loads((DATA / "zurich_world.json").read_text())
    e0, n0 = world["origin"]["east"], world["origin"]["north"]
    grid = world["terrain"]
    half_x = (grid["nx"] - 1) * grid["cell"] / 2
    half_z = (grid["nz"] - 1) * grid["cell"] / 2
    emin, emax = e0 - half_x, e0 + half_x
    nmin, nmax = n0 - half_z, n0 + half_z
    print(f"  world covers E {emin:.0f}..{emax:.0f}, N {nmin:.0f}..{nmax:.0f}")

    urls = tile_urls((47.36, 8.51, 47.39, 8.56))
    print(f"  {len(urls)} CityGML tiles cover it")

    out = []
    for url in urls:
        path = fetch(url)
        kept = scanned = 0
        for text in members(path):
            scanned += 1
            m = FIRST_COORD.search(text)
            if not m:
                continue
            # Cheap rejection before any XML parsing: one coordinate is enough to
            # place a building, and almost every one of them is outside.
            east, north = float(m.group(1)), float(m.group(2))
            if not (emin <= east <= emax and nmin <= north <= nmax):
                continue
            surfaces = parse_member(text)
            if not surfaces:
                continue

            pts = [p for rings in surfaces.values() for r in rings for p in r]
            zs = [p[2] for p in pts]
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            roofs = [[[round(p[0] - e0, 2), round(p[2], 2), round(-(p[1] - n0), 2)]
                      for p in ring] for ring in surfaces["roof"]]
            out.append({
                "x": round(cx - e0, 2),
                "z": round(-(cy - n0), 2),
                "zmin": round(min(zs), 2),
                "zmax": round(max(zs), 2),
                "roof": roofs,
            })
            kept += 1
        print(f"    {path.name[:44]}: {kept} kept of {scanned} scanned", flush=True)

    heights = [b["zmax"] - b["zmin"] for b in out]
    path = DATA / "buildings3d.json"
    path.write_text(json.dumps({"buildings": out}, separators=(",", ":")))
    print(f"\n  {len(out)} measured buildings")
    if heights:
        heights.sort()
        print(f"  heights {heights[0]:.1f}..{heights[-1]:.1f} m, "
              f"median {heights[len(heights)//2]:.1f} m")
    print(f"  wrote {path} ({path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    build()
