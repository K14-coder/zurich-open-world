#!/usr/bin/env python3
"""
Prepare posed 360° panoramas for projective texturing.

This replaces façade reconstruction rather than extending it. Every artifact in
that approach — soft plates, failed registration, a flat fit objective — came
from trying to *rebuild* a flat texture out of many photographs. Projecting one
photograph onto the geometry from its original camera pose needs none of it:
there is only ever a single image in play, at full resolution.

Picks the densest run of panoramas near a chosen point, downloads the originals,
and writes pose plus a rotation matrix per image.
"""

from __future__ import annotations

import io
import json
import math
import pathlib
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from zurich_circuit import wgs84_to_lv95
from zurich_world import terrain_at

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent / "data"
MLY_CACHE = HERE / "cache" / "mapillary"
PANO_DIR = DATA / "panoramas"
PANO_DIR.mkdir(parents=True, exist_ok=True)

# 4096x2048 gives 0.088 deg/px — about 2.3 cm on a wall at 15 m, which is
# comfortably enough to read a shop sign. Full 5660x2830 would be 64 MB each and
# the whole point is to hold a run of them resident at once.
PANO_W, PANO_H = 3072, 1536
COUNT = 44
CAMERA_HEIGHT = 2.2

FIELDS = ("id,thumb_original_url,computed_geometry,computed_rotation,is_pano,"
          "camera_type,captured_at,sequence,width,height")


def token() -> str:
    return (pathlib.Path.home() / ".config" / "mapillary" / "token").read_text().strip()


def rodrigues(vec) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64)
    theta = np.linalg.norm(v)
    if theta < 1e-9:
        return np.eye(3)
    k = v / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def survey_images() -> list:
    images = {}
    for f in MLY_CACHE.glob("*.json"):
        for img in json.loads(f.read_text()).get("data", []):
            if img.get("computed_geometry"):
                images[img["id"]] = img
    return list(images.values())


def detail(image_id: str, tok: str) -> dict | None:
    cache = HERE / "cache" / "pano_meta"
    cache.mkdir(parents=True, exist_ok=True)
    p = cache / f"{image_id}.json"
    if p.exists():
        return json.loads(p.read_text())
    url = (f"https://graph.mapillary.com/{image_id}?"
           + urllib.parse.urlencode({"access_token": tok, "fields": FIELDS}))
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            meta = json.loads(r.read())
        p.write_text(json.dumps(meta))
        return meta
    except Exception:  # noqa: BLE001
        return None


def build(centre_x: float, centre_z: float, radius: float = 320.0) -> None:
    world = json.loads((DATA / "zurich_world.json").read_text())
    e0, n0 = world["origin"]["east"], world["origin"]["north"]
    grid = world["terrain"]
    tok = token()

    # Candidates near the target, projected into the world frame.
    near = []
    for img in survey_images():
        lon, lat = img["computed_geometry"]["coordinates"]
        east, north = wgs84_to_lv95(lat, lon)
        x, z = east - e0, -(north - n0)
        # The survey already carries is_pano, so filter here rather than
        # probing sequences one at a time through the API.
        if img.get("is_pano") and math.hypot(x - centre_x, z - centre_z) <= radius:
            near.append((img, x, z))
    print(f"  {len(near)} images within {radius:.0f} m")

    # One sequence, not a mixture: frames from a single run share camera,
    # exposure and time of day, so transitions between them are not jarring.
    by_seq: dict[str, list] = {}
    for img, x, z in near:
        by_seq.setdefault(img.get("sequence") or "?", []).append((img, x, z))

    ranked = sorted(by_seq.items(), key=lambda kv: -len(kv[1]))
    if not ranked:
        raise SystemExit("no panoramic sequence near that point")
    chosen_seq, frames = ranked[0]
    print(f"  sequence {chosen_seq}: {len(frames)} panoramic frames")

    # Walk the run in capture order and take an evenly spaced subset.
    frames.sort(key=lambda t: t[0].get("captured_at") or 0)
    step = max(1, len(frames) // COUNT)
    frames = frames[::step][:COUNT]

    def grab(item):
        idx, (img, x, z) = item
        meta = detail(img["id"], tok)
        if not meta or not meta.get("computed_rotation"):
            return None
        url = meta.get("thumb_original_url")
        if not url:
            return None
        path = PANO_DIR / f"pano_{idx:02d}.jpg"
        if not path.exists():
            try:
                with urllib.request.urlopen(url, timeout=180) as r:
                    raw = r.read()
                im = Image.open(io.BytesIO(raw)).convert("RGB")
                im = im.resize((PANO_W, PANO_H), Image.LANCZOS)
                im.save(path, "JPEG", quality=92, optimize=True)
            except Exception as exc:  # noqa: BLE001
                print(f"    ! {img['id']}: {exc}")
                return None
        R = rodrigues(meta["computed_rotation"])
        return {
            "file": path.name,
            "pos": [round(x, 3), round(terrain_at(grid, x, z) + CAMERA_HEIGHT, 3),
                    round(z, 3)],
            "R": [round(v, 6) for v in R.flatten().tolist()],
            "captured": meta.get("captured_at"),
        }

    with ThreadPoolExecutor(max_workers=5) as pool:
        results = [r for r in pool.map(grab, list(enumerate(frames))) if r]

    # Order along the run so the renderer can crossfade between neighbours.
    results.sort(key=lambda r: r["captured"] or 0)
    for i, r in enumerate(results):
        r["index"] = i

    (DATA / "panoramas.json").write_text(json.dumps(
        {"width": PANO_W, "height": PANO_H, "panoramas": results},
        separators=(",", ":")))
    span = 0.0
    for a, b in zip(results, results[1:]):
        span += math.dist(a["pos"][::2], b["pos"][::2])
    print(f"  {len(results)} panoramas covering {span:.0f} m of street")
    print(f"  wrote {DATA/'panoramas.json'}")


if __name__ == "__main__":
    import sys
    cx = float(sys.argv[1]) if len(sys.argv) > 2 else 1631.0
    cz = float(sys.argv[2]) if len(sys.argv) > 2 else 609.0
    build(cx, cz)
