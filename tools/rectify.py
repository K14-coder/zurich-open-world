#!/usr/bin/env python3
"""
Rectify Mapillary street imagery onto façade planes and composite clean plates.

For each façade this walks a grid of points across the wall, projects each point
back into every photograph that can see it, and samples there. Inverse mapping
rather than forward: projecting pixels onto the wall leaves holes wherever the
wall is sampled sparsely, whereas asking "what colour is this bit of wall?"
cannot.

Cars, bikes and people are then removed by taking the per-pixel median across
views, which the building wins because it is the only thing that did not move
between passes.

    python3 rectify.py --street Freiestrasse --limit 8
"""

from __future__ import annotations

import argparse
import io
import json
import math
import pathlib
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from zurich_circuit import wgs84_to_lv95
from zurich_world import terrain_at
from clean_plate import median_composite

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent / "data"
MLY_CACHE = HERE / "cache" / "mapillary"
IMG_CACHE = HERE / "cache" / "mly_images"
IMG_CACHE.mkdir(parents=True, exist_ok=True)

PIXELS_PER_METRE = 20          # 5 cm/px
MAX_RANGE = 30.0
MAX_OFF_AXIS = math.radians(65)
# Camera height above the road. Mapillary's computed_altitude is WGS84
# ellipsoidal, which sits ~50 m off orthometric height in Switzerland; getting
# that wrong shears the whole façade vertically, so a fixed height above our own
# terrain is both simpler and far more robust.
CAMERA_HEIGHT = 2.2

FIELDS = ("id,thumb_2048_url,computed_geometry,computed_rotation,camera_type,"
          "camera_parameters,is_pano,captured_at,sequence,width,height")


def token() -> str:
    return (pathlib.Path.home() / ".config" / "mapillary" / "token").read_text().strip()


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def rodrigues(vec) -> np.ndarray:
    """Angle-axis to rotation matrix. Mapillary gives camera orientation this way."""
    v = np.asarray(vec, dtype=np.float64)
    theta = np.linalg.norm(v)
    if theta < 1e-9:
        return np.eye(3)
    k = v / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def project(cam_pts: np.ndarray, meta: dict) -> tuple[np.ndarray, np.ndarray]:
    """Camera-space points to pixel coordinates. Returns pixels and a validity
    mask (points behind the camera or outside the frame are invalid)."""
    w, h = meta["width"], meta["height"]
    X, Y, Z = cam_pts[..., 0], cam_pts[..., 1], cam_pts[..., 2]

    if meta.get("is_pano") or meta.get("camera_type") == "spherical":
        # Equirectangular: every direction is in frame, so only degenerate
        # points are rejected.
        lon = np.arctan2(X, Z)
        lat = np.arctan2(-Y, np.hypot(X, Z))
        px = (0.5 + lon / (2 * math.pi)) * w
        py = (0.5 - lat / math.pi) * h
        valid = np.isfinite(px) & np.isfinite(py)
        return np.stack([px, py], axis=-1), valid

    params = meta.get("camera_parameters") or [0.85, 0.0, 0.0]
    focal = params[0] if params else 0.85
    k1 = params[1] if len(params) > 1 else 0.0
    k2 = params[2] if len(params) > 2 else 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        x = X / Z
        y = Y / Z
    r2 = x * x + y * y
    d = 1 + k1 * r2 + k2 * r2 * r2
    scale = max(w, h)
    px = focal * d * x * scale + w / 2
    py = focal * d * y * scale + h / 2
    valid = (Z > 0.1) & np.isfinite(px) & np.isfinite(py)
    valid &= (px >= 0) & (px < w - 1) & (py >= 0) & (py < h - 1)
    return np.stack([px, py], axis=-1), valid


def sample_bilinear(img: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    h, w, _ = img.shape
    # Invalid projections arrive as NaN; casting those to int is undefined, so
    # neutralise them here and let the validity mask discard them later.
    px = np.nan_to_num(px, nan=0.0, posinf=0.0, neginf=0.0)
    py = np.nan_to_num(py, nan=0.0, posinf=0.0, neginf=0.0)
    x0 = np.clip(np.floor(px).astype(np.int32), 0, w - 2)
    y0 = np.clip(np.floor(py).astype(np.int32), 0, h - 2)
    fx = np.clip(px - x0, 0, 1)[..., None]
    fy = np.clip(py - y0, 0, 1)[..., None]
    a = img[y0, x0] * (1 - fx) * (1 - fy)
    b = img[y0, x0 + 1] * fx * (1 - fy)
    c = img[y0 + 1, x0] * (1 - fx) * fy
    d = img[y0 + 1, x0 + 1] * fx * fy
    return a + b + c + d


# ---------------------------------------------------------------------------
# Imagery
# ---------------------------------------------------------------------------

def cached_survey_images() -> list:
    images = {}
    for f in MLY_CACHE.glob("*.json"):
        for img in json.loads(f.read_text()).get("data", []):
            if img.get("computed_geometry"):
                images[img["id"]] = img
    return list(images.values())


def fetch_details(ids: list[str], tok: str) -> dict:
    out = {}
    todo = []
    for i in ids:
        p = IMG_CACHE / f"{i}.json"
        if p.exists():
            out[i] = json.loads(p.read_text())
        else:
            todo.append(i)
    for i in todo:
        url = (f"https://graph.mapillary.com/{i}?"
               + urllib.parse.urlencode({"access_token": tok, "fields": FIELDS}))
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                meta = json.loads(r.read())
            (IMG_CACHE / f"{i}.json").write_text(json.dumps(meta))
            out[i] = meta
        except Exception as exc:  # noqa: BLE001
            print(f"    ! detail {i}: {exc}", file=sys.stderr)
    return out


def fetch_pixels(meta: dict) -> np.ndarray | None:
    path = IMG_CACHE / f"{meta['id']}.jpg"
    if not path.exists():
        url = meta.get("thumb_2048_url")
        if not url:
            return None
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                path.write_bytes(r.read())
        except Exception as exc:  # noqa: BLE001
            print(f"    ! image {meta['id']}: {exc}", file=sys.stderr)
            return None
    try:
        img = Image.open(path).convert("RGB")
    except Exception:  # noqa: BLE001
        return None
    # The thumbnail is not the full-resolution frame the intrinsics describe, so
    # rescale the stored dimensions to match what we actually sampled.
    meta["width"], meta["height"] = img.size
    return np.asarray(img).astype(np.float32)



# ---------------------------------------------------------------------------
# Visibility
# ---------------------------------------------------------------------------

def build_occluders(buildings: list, cell: float = 40.0):
    """Grid of building footprints, for testing whether a camera can actually
    see a wall. Without this, any camera within range pointing roughly the right
    way is accepted — including ones photographing the building opposite."""
    index: dict[tuple[int, int], list] = {}
    for bid, b in enumerate(buildings):
        ring = [(p[0], p[1]) for p in b["r"]]
        if len(ring) < 3:
            continue
        xs = [p[0] for p in ring]
        zs = [p[1] for p in ring]
        entry = (bid, ring, min(xs), max(xs), min(zs), max(zs))
        for i in range(int(min(xs) // cell), int(max(xs) // cell) + 1):
            for j in range(int(min(zs) // cell), int(max(zs) // cell) + 1):
                index.setdefault((i, j), []).append(entry)
    return (index, cell)


def _inside(px, pz, ring) -> bool:
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


def blocked(cx, cz, tx, tz, own_id, occluders) -> bool:
    """True if any other building's footprint lies between camera and wall."""
    index, cell = occluders
    dist = math.hypot(tx - cx, tz - cz)
    steps = max(2, int(dist / 2.0))
    # Stop short of the target so the wall's own building never blocks itself.
    for s in range(1, steps):
        t = s / steps
        px = cx + (tx - cx) * t
        pz = cz + (tz - cz) * t
        if math.hypot(tx - px, tz - pz) < 1.5:
            break
        for (bid, ring, x0, x1, z0, z1) in index.get((int(px // cell), int(pz // cell)), []):
            if bid == own_id or not (x0 <= px <= x1 and z0 <= pz <= z1):
                continue
            if _inside(px, pz, ring):
                return True
    return False


def looks_like_sky(view, mask) -> float:
    """Fraction of valid pixels that are probably sky. Building heights are
    estimated rather than measured, so an over-tall façade samples sky across its
    upper storeys and poisons the median."""
    if mask.sum() == 0:
        return 1.0
    v = view[mask].astype(np.float32)
    if v.size == 0:
        return 1.0
    blue = (v[:, 2] > v[:, 0] + 12) & (v[:, 2] > 90)
    return float(blue.mean())


# ---------------------------------------------------------------------------
# Rectification
# ---------------------------------------------------------------------------

def rectify_facade(f: dict, images: list, grid: dict, e0: float, n0: float,
                   tok: str, occluders=None, max_views: int = 14):
    ax, az = f["a"]
    bx, bz = f["z"]
    nx, nz = f["n"]
    length = math.hypot(bx - ax, bz - az)
    base, height = f["base"], f["h"]

    # Candidate views: near enough, on the right side, pointing at the wall.
    mx, mz = (ax + bx) / 2, (az + bz) / 2
    cands = []
    for img in images:
        x, z, dx, dz, seq = img["_p"]
        vx, vz = mx - x, mz - z
        dist = math.hypot(vx, vz)
        if dist > MAX_RANGE or dist < 1.0:
            continue
        if (x - mx) * nx + (z - mz) * nz <= 0:
            continue
        if (vx * dx + vz * dz) / dist < math.cos(MAX_OFF_AXIS):
            continue
        if occluders is not None and blocked(x, z, mx, mz, f["b"], occluders):
            continue
        cands.append((dist, img))
    if not cands:
        return None, 0

    # One view per sequence, nearest wins: extra frames from the same run carry
    # the same parked cars and would stack the vote.
    by_seq: dict[str, tuple] = {}
    for dist, img in sorted(cands, key=lambda c: c[0]):
        by_seq.setdefault(img["_p"][4], (dist, img))
    chosen = [img for _, img in list(by_seq.values())[:max_views]]
    if len(chosen) < 2:
        return None, len(chosen)

    details = fetch_details([c["id"] for c in chosen], tok)

    # Sample grid across the wall, in world space.
    W = max(24, int(length * PIXELS_PER_METRE))
    H = max(24, int(height * PIXELS_PER_METRE))
    u = (np.arange(W) + 0.5) / W
    v = (np.arange(H) + 0.5) / H
    U, V = np.meshgrid(u, v)
    Px = ax + (bx - ax) * U
    Pz = az + (bz - az) * U
    Py = base + height * (1.0 - V)          # v=0 is the top of the wall

    views, masks = [], []
    for img in chosen:
        meta = details.get(img["id"])
        if not meta or not meta.get("computed_rotation"):
            continue
        pixels = fetch_pixels(meta)
        if pixels is None:
            continue

        lon, lat = meta["computed_geometry"]["coordinates"]
        east, north = wgs84_to_lv95(lat, lon)
        cx, cz = east - e0, -(north - n0)
        cy = terrain_at(grid, cx, cz) + CAMERA_HEIGHT

        # Our world is X east, Y up, Z south. Mapillary's rotation is expressed
        # against local ENU, so swap into (east, north, up) before rotating.
        dX, dY, dZ = Px - cx, Py - cy, Pz - cz
        enu = np.stack([dX, -dZ, dY], axis=-1)
        R = rodrigues(meta["computed_rotation"])
        cam = enu @ R.T

        px_py, valid = project(cam, meta)
        sampled = sample_bilinear(pixels, px_py[..., 0], px_py[..., 1])
        sampled = np.where(valid[..., None], sampled, 0.0)
        if valid.mean() < 0.5:
            continue
        shot = np.clip(sampled, 0, 255).astype(np.uint8)
        if looks_like_sky(shot, valid) > 0.35:
            continue
        views.append(shot)
        masks.append(valid)

    if len(views) < 2:
        return None, len(views)

    raw_first = views[0]
    views, masks, aligned = align_views(views, masks)
    plate = median_composite(views, masks)
    return (plate, raw_first, aligned), len(views)


def phase_shift(ref: np.ndarray, img: np.ndarray, max_shift: float = 0.18):
    """Integer (dy, dx) that best aligns `img` onto `ref`, by phase correlation.

    Rectification alone does not put two views in register: an OSM footprint sits
    a metre or two off the true wall plane, GPS is metres out, and the assumed
    camera height adds its own error. Each view therefore lands slightly shifted,
    and a median over unregistered views smears rather than cleans.
    """
    a = ref.mean(axis=2) - ref.mean()
    b = img.mean(axis=2) - img.mean()
    if a.std() < 1e-3 or b.std() < 1e-3:
        return 0, 0
    A = np.fft.rfft2(a)
    B = np.fft.rfft2(b)
    cross = A * np.conj(B)
    mag = np.abs(cross)
    mag[mag < 1e-9] = 1e-9
    corr = np.fft.irfft2(cross / mag, s=a.shape)

    h, w = a.shape
    limit_y = max(1, int(h * max_shift))
    limit_x = max(1, int(w * max_shift))
    # Only consider shifts inside the plausible band, wrapped both ways.
    ys = np.r_[0:limit_y, h - limit_y:h]
    xs = np.r_[0:limit_x, w - limit_x:w]
    window = corr[np.ix_(ys, xs)]
    idx = np.unravel_index(np.argmax(window), window.shape)
    dy, dx = ys[idx[0]], xs[idx[1]]
    if dy > h // 2:
        dy -= h
    if dx > w // 2:
        dx -= w
    return int(dy), int(dx)


def align_views(views: list, masks: list):
    """Register every view onto the one with the most valid coverage.

    A translation is not enough. Each view sees the wall from a different
    distance and angle, and the façade plane taken from an OSM footprint is only
    approximately where the real wall is, so the residual between two rectified
    views is a homography — the reason a plain median came out as a smear.
    ORB features plus RANSAC recover it.
    """
    import cv2

    ref_i = int(np.argmax([m.mean() for m in masks]))
    ref = views[ref_i]
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
    h, w = ref_gray.shape

    orb = cv2.ORB_create(nfeatures=1500)
    kp_ref, des_ref = orb.detectAndCompute(ref_gray, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    out_v, out_m, aligned = [], [], 0
    for i, (v, m) in enumerate(zip(views, masks)):
        if i == ref_i or des_ref is None or len(kp_ref) < 12:
            out_v.append(v); out_m.append(m)
            continue

        gray = cv2.cvtColor(v, cv2.COLOR_RGB2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        if des is None or len(kp) < 12:
            out_v.append(v); out_m.append(m)
            continue

        matches = sorted(matcher.match(des, des_ref), key=lambda x: x.distance)[:220]
        if len(matches) < 12:
            out_v.append(v); out_m.append(m)
            continue

        src = np.float32([kp[x.queryIdx].pt for x in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp_ref[x.trainIdx].pt for x in matches]).reshape(-1, 1, 2)
        H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None or inliers is None or int(inliers.sum()) < 10:
            out_v.append(v); out_m.append(m)
            continue

        # Reject anything that is not a mild correction: a wild homography means
        # the match latched onto a repeating window pattern, which façades are
        # full of, and warping by it would be worse than leaving the view alone.
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        moved = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        if np.abs(moved - corners.reshape(-1, 2)).max() > 0.25 * max(w, h):
            out_v.append(v); out_m.append(m)
            continue

        out_v.append(cv2.warpPerspective(v, H, (w, h), flags=cv2.INTER_LINEAR))
        out_m.append(cv2.warpPerspective(m.astype(np.uint8), H, (w, h),
                                         flags=cv2.INTER_NEAREST).astype(bool))
        aligned += 1

    return out_v, out_m, aligned


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--street", default="Freiestrasse")
    ap.add_argument("--limit", type=int, default=8)
    args = ap.parse_args()

    tok = token()
    world = json.loads((DATA / "zurich_world.json").read_text())
    e0, n0 = world["origin"]["east"], world["origin"]["north"]
    grid = world["terrain"]
    facades = json.loads((DATA / "zurich_facades.json").read_text())["facades"]

    targets = sorted(
        (f for f in facades if f["road"] == args.street and f.get("passes", 0) >= 4),
        key=lambda f: -f["passes"])[:args.limit]
    if not targets:
        raise SystemExit(f"no well-covered façades on {args.street}")
    print(f"  {len(targets)} façades on {args.street}, "
          f"{targets[0]['passes']}-{targets[-1]['passes']} passes each")

    buildings = json.loads((DATA / "zurich_buildings.json").read_text())["buildings"]
    occluders = build_occluders(buildings)
    print(f"  built occlusion index from {len(buildings)} footprints")

    images = cached_survey_images()
    for img in images:
        lon, lat = img["computed_geometry"]["coordinates"]
        east, north = wgs84_to_lv95(lat, lon)
        a = math.radians(img.get("compass_angle") or 0.0)
        img["_p"] = (east - e0, -(north - n0), math.sin(a), -math.cos(a),
                     img.get("sequence"))
    print(f"  {len(images)} surveyed images available")

    out_dir = HERE.parent / "images" / "facades"
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []

    for i, f in enumerate(targets):
        result, n = rectify_facade(f, images, grid, e0, n0, tok, occluders)
        if result is None:
            print(f"    façade {i}: only {n} usable views, skipped")
            continue
        plate, raw_first, aligned = result
        Image.fromarray(plate).save(out_dir / f"facade_{i:02d}_plate.jpg", quality=92)
        # Side by side with a single raw view, so the effect is visible.
        strip = Image.new("RGB", (plate.shape[1] * 2, plate.shape[0]))
        strip.paste(Image.fromarray(raw_first), (0, 0))
        strip.paste(Image.fromarray(plate), (plate.shape[1], 0))
        strip.save(out_dir / f"facade_{i:02d}_compare.jpg", quality=92)
        made.append((i, f, n))
        print(f"    façade {i}: {n} views ({aligned} homography-aligned) -> "
              f"{plate.shape[1]}x{plate.shape[0]} plate  ({f['len']:.1f} m wide)")

    if made:
        print(f"\n  wrote {len(made)} plates to {out_dir}")
    else:
        print("\n  no plates produced")


if __name__ == "__main__":
    main()
