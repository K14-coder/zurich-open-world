#!/usr/bin/env python3
"""
Build an ApexSim TrackSpec for a street circuit through central Zurich from open data.

Sources
  * OpenStreetMap (ODbL) via Overpass — road centrelines, lane counts, widths.
  * swisstopo swissALTI3D via the geo.admin height API — real elevation.

The circuit is an ordered list of real streets. Each street's OSM ways are
stitched into a single polyline, the streets are chained end to end, the loop is
closed, simplified to control points, and elevation is sampled per point.

Output is a Swift source file declaring a TrackSpec, plus a GeoJSON for eyeballing
the route on a map before trusting it.
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

# Swiss Overpass mirror first — closest to the data and reliably up.
OVERPASS = [
    "https://overpass.osm.ch/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]
HEIGHT_API = "https://api3.geo.admin.ch/rest/services/height"

# The lap, in order. A real closed loop through the middle of Zurich:
# down Bahnhofstrasse to the lake, east over the Quaibrücke to Bellevue, back
# north along the Limmat on Limmatquai, over the Bahnhofbrücke and down the
# Bahnhofquai to where we started.
CIRCUIT = ["Bahnhofstrasse", "Quaibrücke", "Limmatquai", "Bahnhofbrücke", "Bahnhofquai"]
BBOX = (47.3640, 8.5330, 47.3790, 8.5490)  # south, west, north, east

CACHE = pathlib.Path(__file__).with_name("cache")
CACHE.mkdir(exist_ok=True)


# --------------------------------------------------------------------------
# Coordinates
# --------------------------------------------------------------------------

def wgs84_to_lv95(lat: float, lon: float) -> tuple[float, float]:
    """swisstopo's approximate WGS84 -> LV95 formula. Good to about a metre,
    which is far below the width of a lane."""
    phi = lat * 3600.0
    lam = lon * 3600.0
    p = (phi - 169028.66) / 10000.0
    l = (lam - 26782.5) / 10000.0
    east = (
        2600072.37
        + 211455.93 * l
        - 10938.51 * l * p
        - 0.36 * l * p * p
        - 44.54 * l**3
    )
    north = (
        1200147.07
        + 308807.95 * p
        + 3745.25 * l * l
        + 76.63 * p * p
        - 194.56 * l * l * p
        + 119.79 * p**3
    )
    return east, north


# --------------------------------------------------------------------------
# Overpass
# --------------------------------------------------------------------------

def overpass(query: str) -> dict:
    key = CACHE / f"ovp_{abs(hash(query))}.json"
    if key.exists():
        return json.loads(key.read_text())
    last = None
    for endpoint in OVERPASS:
        try:
            data = urllib.parse.urlencode({"data": query}).encode()
            req = urllib.request.Request(endpoint, data=data)
            with urllib.request.urlopen(req, timeout=90) as resp:
                out = json.loads(resp.read().decode())
            key.write_text(json.dumps(out))
            return out
        except Exception as exc:  # noqa: BLE001 - mirrors fail routinely
            last = exc
            print(f"  ! {endpoint} failed: {exc}", file=sys.stderr)
            time.sleep(2)
    raise RuntimeError(f"every Overpass mirror failed: {last}")


# Road classes that form a street corridor. `service` is deliberately excluded:
# courtyards and delivery spurs share the street's name, and because they loop
# back on themselves they derail any attempt to walk the street end to end.
# `pedestrian` is kept — the length of Bahnhofstrasse is tram and foot traffic in
# real life, and it is the whole point of the lap.
ROAD_CLASSES = (
    "motorway|trunk|primary|secondary|tertiary|residential|unclassified"
    "|living_street|pedestrian"
)


def fetch_street(name: str) -> list[dict]:
    s, w, n, e = BBOX
    query = (
        f"[out:json][timeout:60];"
        f'way["name"="{name}"]["highway"~"^({ROAD_CLASSES})$"]({s},{w},{n},{e});'
        f"out body geom;"
    )
    return [el for el in overpass(query).get("elements", []) if el.get("type") == "way"]


# --------------------------------------------------------------------------
# Stitching ways into one polyline
# --------------------------------------------------------------------------

def stitch(ways: list[dict]) -> list[tuple[float, float]]:
    """Greedily merge ways that share an endpoint node into chains, then return
    the longest chain's geometry. OSM splits a street into many ways wherever a
    tag changes, so this reassembly is the bulk of the work."""
    segs = []
    for way in ways:
        nodes, geom = way.get("nodes", []), way.get("geometry", [])
        if len(nodes) < 2 or len(geom) != len(nodes):
            continue
        # A closed way is a pedestrian plaza mapped as an area, not a stretch of
        # road. Bahnhofstrasse's traffic-free section is drawn this way. Walking
        # one produces a kilometre of geometry that goes precisely nowhere.
        if nodes[0] == nodes[-1]:
            continue
        segs.append((list(nodes), [(p["lat"], p["lon"]) for p in geom]))
    if not segs:
        return []

    # Index every way by both of its endpoint nodes, then grow a chain outwards
    # from a seed way in both directions. Growing from both ends matters: seeding
    # in the middle of a street and only ever appending would strand half of it.
    by_end: dict[int, list[int]] = {}
    for i, (nodes, _) in enumerate(segs):
        by_end.setdefault(nodes[0], []).append(i)
        by_end.setdefault(nodes[-1], []).append(i)

    used = [False] * len(segs)
    chains = []

    for seed in range(len(segs)):
        if used[seed]:
            continue
        used[seed] = True
        nodes, geom = list(segs[seed][0]), list(segs[seed][1])

        for _ in range(2):  # once forward off the tail, once backward off the head
            while True:
                tail = nodes[-1]
                nxt = next((i for i in by_end.get(tail, []) if not used[i]), None)
                if nxt is None:
                    break
                used[nxt] = True
                n2, g2 = segs[nxt]
                if n2[0] == tail:
                    nodes.extend(n2[1:]); geom.extend(g2[1:])
                else:
                    nodes.extend(n2[::-1][1:]); geom.extend(g2[::-1][1:])
            nodes.reverse(); geom.reverse()  # flip and grow the other way

        chains.append((nodes, geom))

    # Dropping the plazas leaves real holes in the corridor, because the plaza was
    # carrying the street across that stretch. Rejoin chains whose ends are close
    # enough that only a plaza or a junction can be sitting between them.
    geoms = merge_by_proximity([g for _, g in chains], max_gap=160.0)

    # Pick the chain that reaches furthest end to end, not the one with the most
    # tarmac. A street that loops back on itself can have huge arc length while
    # going nowhere, and that is exactly the shape we must not pick.
    return max(geoms, key=lambda g: haversine(g[0], g[-1]))


def merge_by_proximity(geoms: list[list], max_gap: float) -> list[list]:
    """Repeatedly join the two closest chain endpoints while they are within
    `max_gap` metres, flipping chains as needed so the join is end-to-start."""
    geoms = [list(g) for g in geoms if len(g) >= 2]
    while len(geoms) > 1:
        best = None
        for i in range(len(geoms)):
            for j in range(i + 1, len(geoms)):
                for ei in (0, -1):
                    for ej in (0, -1):
                        d = haversine(geoms[i][ei], geoms[j][ej])
                        if d < max_gap and (best is None or d < best[0]):
                            best = (d, i, j, ei, ej)
        if best is None:
            break
        _, i, j, ei, ej = best
        a = geoms[i][::-1] if ei == 0 else geoms[i]
        b = geoms[j][::-1] if ej == -1 else geoms[j]
        geoms[i] = a + b
        geoms.pop(j)
    return geoms


def haversine(a: tuple[float, float], b: tuple[float, float]) -> float:
    r = 6371000.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp = p2 - p1
    dl = math.radians(b[1] - a[1])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def closest_pair(a: list, b: list) -> tuple[int, int, float]:
    """Index into `a`, index into `b`, and the distance between them, for the
    closest pair of vertices. This is where two streets meet."""
    best = (0, 0, float("inf"))
    for i, pa in enumerate(a):
        for j, pb in enumerate(b):
            d = haversine(pa[:2], pb[:2])
            if d < best[2]:
                best = (i, j, d)
    return best


def polyline_length(pts: list[tuple[float, float]]) -> float:
    return sum(haversine(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def road_width(ways: list[dict]) -> float:
    """Prefer an explicit width tag, else lanes x 3.25 m, else a sane default."""
    widths = []
    for way in ways:
        tags = way.get("tags", {})
        if "width" in tags:
            try:
                widths.append(float(str(tags["width"]).split()[0]))
                continue
            except ValueError:
                pass
        if "lanes" in tags:
            try:
                widths.append(int(tags["lanes"]) * 3.25)
            except ValueError:
                pass
    if not widths:
        return 11.0
    return min(20.0, max(8.0, sorted(widths)[len(widths) // 2]))


# --------------------------------------------------------------------------
# Elevation
# --------------------------------------------------------------------------

def height_at(east: float, north: float) -> float:
    url = f"{HEIGHT_API}?easting={east:.1f}&northing={north:.1f}&sr=2056"
    for _ in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return float(json.loads(resp.read().decode())["height"])
        except Exception:  # noqa: BLE001
            time.sleep(1.0)
    return float("nan")


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def simplify(pts: list[tuple], tol: float) -> list[tuple]:
    """Douglas-Peucker on (east, north) metres."""
    if len(pts) < 3:
        return pts
    dmax, idx = 0.0, 0
    a, b = pts[0], pts[-1]
    for i in range(1, len(pts) - 1):
        d = point_line_distance(pts[i], a, b)
        if d > dmax:
            dmax, idx = d, i
    if dmax > tol:
        return simplify(pts[: idx + 1], tol)[:-1] + simplify(pts[idx:], tol)
    return [a, b]


def point_line_distance(p, a, b) -> float:
    if a[:2] == b[:2]:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    num = abs((b[0] - a[0]) * (a[1] - p[1]) - (a[0] - p[0]) * (b[1] - a[1]))
    return num / math.hypot(b[0] - a[0], b[1] - a[1])


def segments_intersect(p1, p2, p3, p4) -> bool:
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1, d2 = cross(p3, p4, p1), cross(p3, p4, p2)
    d3, d4 = cross(p1, p2, p3), cross(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def self_intersects(pts: list[tuple]) -> list[tuple[int, int]]:
    """The polar authoring form guaranteed this could never happen. Real streets
    carry no such guarantee, so the generator has to check it: a circuit that
    crosses itself breaks lap timing and the racing-line solver."""
    hits = []
    n = len(pts)
    for i in range(n):
        a1, a2 = pts[i], pts[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            b1, b2 = pts[j], pts[(j + 1) % n]
            if segments_intersect(a1, a2, b1, b2):
                hits.append((i, j))
    return hits


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

def build() -> None:
    print("Fetching streets from OpenStreetMap...")
    streets: list[tuple[str, list[tuple[float, float]], float]] = []

    for name in CIRCUIT:
        ways = fetch_street(name)
        pts = stitch(ways)
        if not pts:
            print(f"  ! {name}: no geometry found", file=sys.stderr)
            continue
        width = road_width(ways)
        streets.append((name, pts, width))
        print(f"  {name}: {len(ways)} ways -> {len(pts)} pts, "
              f"{polyline_length(pts):.0f} m, width {width:.1f} m")

    # An OSM street runs well past the junctions that bound our lap, and nothing
    # fixes which end of the first street we start from. Both problems are the
    # same problem: find where consecutive streets actually meet, and keep only
    # the stretch between one junction and the next. That also fixes orientation
    # for free, including for the first street.
    print("\nTrimming streets at their junctions...")
    n = len(streets)
    junctions = [closest_pair(streets[k][1], streets[(k + 1) % n][1]) for k in range(n)]

    chain: list[tuple[float, float]] = []
    widths: list[tuple[int, float]] = []
    for k, (name, pts, width) in enumerate(streets):
        enter = junctions[k - 1][1]        # where the previous street handed over
        leave = junctions[k][0]            # where we hand over to the next
        seg = pts[enter:leave + 1] if enter <= leave else pts[leave:enter + 1][::-1]
        if len(seg) < 2:
            print(f"  ! {name}: junctions collapsed to nothing, keeping whole street",
                  file=sys.stderr)
            seg = pts
        gap = junctions[k][2]
        flag = "  <-- large gap" if gap > 60 else ""
        print(f"  {name}: kept {len(seg)}/{len(pts)} pts, "
              f"{polyline_length(seg):.0f} m, junction gap {gap:.0f} m{flag}")
        widths.append((len(chain), width))
        chain.extend(seg)

    if len(chain) < 8:
        raise SystemExit("not enough geometry to build a circuit")

    # Project to LV95 metres and re-origin on the circuit's centroid.
    proj = [wgs84_to_lv95(lat, lon) for lat, lon in chain]
    e0 = sum(p[0] for p in proj) / len(proj)
    n0 = sum(p[1] for p in proj) / len(proj)
    # Game frame: X east, Z south (so north is -Z), Y up.
    local = [(e - e0, -(n - n0)) for e, n in proj]

    idx_width = {i: w for i, w in widths}
    tagged = []
    current = 11.0
    for i, (x, z) in enumerate(local):
        current = idx_width.get(i, current)
        tagged.append((x, z, current))

    pts = simplify(tagged, tol=3.0)
    print(f"\nSimplified {len(tagged)} -> {len(pts)} control points")

    crossings = self_intersects([(p[0], p[1]) for p in pts])
    if crossings:
        print(f"  ! circuit self-intersects at {len(crossings)} place(s): {crossings[:5]}",
              file=sys.stderr)
    else:
        print("  self-intersection check: clean")

    print("Sampling swissALTI3D elevation...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        heights = list(pool.map(
            lambda p: height_at(p[0] + e0, n0 - p[1]), pts
        ))
    valid = [h for h in heights if not math.isnan(h)]
    if not valid:
        raise SystemExit("elevation lookup failed for every point")
    base = min(valid)
    heights = [(base if math.isnan(h) else h) - base for h in heights]
    print(f"  elevation range {min(heights):.1f} to {max(heights):.1f} m "
          f"(datum {base:.1f} m)")

    length = sum(
        math.dist(pts[i][:2], pts[(i + 1) % len(pts)][:2]) for i in range(len(pts))
    )
    print(f"  lap length {length:.0f} m")

    write_swift(pts, heights, length)
    write_geojson(chain)


def write_swift(pts, heights, length) -> None:
    lines = [
        "import Foundation",
        "",
        "/// Zurich — a street circuit built from real geometry, not authored by hand.",
        "///",
        "/// The centreline is OpenStreetMap road data for Bahnhofstrasse, the",
        "/// Quaibrücke, Limmatquai, the Bahnhofbrücke and the Bahnhofquai, stitched",
        "/// into a closed lap. Widths come from OSM lane counts and the elevation",
        "/// profile is sampled from swisstopo's swissALTI3D terrain model, so the",
        "/// climb away from the lake is the one that is actually there.",
        "///",
        "/// Unlike the authored circuits this is Cartesian rather than polar, because",
        "/// a real street loop is not star-shaped about any centre. The generator",
        f"/// checks it for self-intersection instead. Lap length {length:.0f} m.",
        "///",
        "/// Generated by Tools/zurich_circuit.py — edit that, not this file.",
        "/// Road data (c) OpenStreetMap contributors, ODbL. Elevation (c) swisstopo.",
        "extension TrackLibrary {",
        "    public static let zurich = TrackSpec(",
        '        id: "zurich",',
        '        name: "Zurich Street Circuit",',
        '        location: "Zurich, Switzerland",',
        '        blurb: "Down Bahnhofstrasse to the lake, over the Quaibrücke and back '
        'along the Limmat. Narrow, walled and unforgiving.",',
        "        controlPoints: [",
    ]
    for (x, z, w), y in zip(pts, heights):
        lines.append(
            f"            TrackControlPoint({x:9.2f}, {y:6.2f}, {z:9.2f}, width: {w:.1f}),"
        )
    lines += [
        "        ],",
        "        kerbWidth: 0.6,",
        "        sectorSplits: [0.333, 0.666]",
        "    )",
        "}",
        "",
    ]
    out = pathlib.Path(__file__).with_name("ZurichCircuit.swift")
    out.write_text("\n".join(lines))
    print(f"\nWrote {out}")


def write_geojson(chain) -> None:
    out = pathlib.Path(__file__).with_name("zurich_circuit.geojson")
    out.write_text(json.dumps({
        "type": "Feature",
        "properties": {"name": "Zurich Street Circuit"},
        "geometry": {
            "type": "LineString",
            "coordinates": [[lon, lat] for lat, lon in chain],
        },
    }))
    print(f"Wrote {out}  (drag onto geojson.io to check the route)")


if __name__ == "__main__":
    build()
