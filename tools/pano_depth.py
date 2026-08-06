#!/usr/bin/env python3
"""
Give each panorama its own depth, so the photograph can be the geometry.

Projecting photographs onto a model of the city has a ceiling that no amount of
better modelling raises: the model contains buildings, and the photograph
contains trees, poles, awnings, balconies, parked cars and signs. Everything the
model lacks gets smeared onto whatever box is nearest.

With per-pixel depth the photograph becomes its own geometry. Every object in it
is represented because every pixel knows its distance, and a viewpoint near the
capture point sees correct parallax — which is all a driving game needs, since
the panoramas were shot from a car on the same road the player drives.

Depth Anything V2 is trained on perspective images, not equirectangular ones, so
each panorama is cut into cubemap faces, estimated separately, and reassembled.
The model returns *relative* inverse depth, so it is then anchored to metres
against the one distance we already know: the road surface, which sits at
camera height divided by the sine of the angle below the horizon.

    python3 pano_depth.py --one    estimate a single panorama and write a preview
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.request

import numpy as np
import onnxruntime as ort
from PIL import Image

from zurich_world import terrain_at

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent / "data"
PANOS = DATA / "panoramas"
MODEL = pathlib.Path.home() / ".cache" / "zurich-models" / "depth_anything_v2_small.onnx"
MODEL_URL = ("https://huggingface.co/onnx-community/depth-anything-v2-small/"
             "resolve/main/onnx/model.onnx")

FACE = 518          # Depth Anything V2 input size
# Faces are cut at 110°, not 90°. At exactly 90° adjacent faces only touch along
# an edge, and the joint solve below needs shared directions — genuinely
# overlapping cones — to tie one face's scale to its neighbour's.
FACE_FOV = 110.0
CAMERA_HEIGHT = 2.2
# Faces: name, forward vector. Up is omitted — it is sky, and sky has no useful
# depth; down is kept because the road is the anchor the scale is fitted to.
FACES = [
    ("front", np.array([0, 0, 1.0])),
    ("right", np.array([1.0, 0, 0])),
    ("back", np.array([0, 0, -1.0])),
    ("left", np.array([-1.0, 0, 0])),
    ("down", np.array([0, 1.0, 0])),
]

_session = None


def session() -> ort.InferenceSession:
    global _session
    if _session is None:
        if not MODEL.exists():
            MODEL.parent.mkdir(parents=True, exist_ok=True)
            print(f"  fetching depth model -> {MODEL}", flush=True)
            tmp = MODEL.with_suffix(".part")
            with urllib.request.urlopen(MODEL_URL, timeout=1800) as r:
                tmp.write_bytes(r.read())
            tmp.rename(MODEL)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _session = ort.InferenceSession(str(MODEL), opts,
                                        providers=["CPUExecutionProvider"])
    return _session


def face_directions(forward: np.ndarray, size: int) -> np.ndarray:
    """Unit ray per pixel of a 90° perspective face."""
    up_hint = np.array([0, 1.0, 0])
    if abs(np.dot(forward, up_hint)) > 0.9:
        up_hint = np.array([0, 0, 1.0])
    right = np.cross(up_hint, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)

    half = np.tan(np.radians(FACE_FOV) / 2)
    t = ((np.arange(size) + 0.5) / size * 2 - 1) * half
    u, v = np.meshgrid(t, t)
    d = forward[None, None, :] + u[..., None] * right + v[..., None] * up
    return d / np.linalg.norm(d, axis=2, keepdims=True)


def sample_equirect(img: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    lon = np.arctan2(x, z)
    lat = np.arctan2(-y, np.hypot(x, z))
    px = ((0.5 + lon / (2 * np.pi)) * w).astype(np.int32) % w
    py = np.clip(((0.5 - lat / np.pi) * h).astype(np.int32), 0, h - 1)
    return img[py, px]


def estimate(face: np.ndarray) -> np.ndarray:
    """Relative inverse depth for one perspective face."""
    x = face.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))[None]
    out = session().run(None, {session().get_inputs()[0].name: x})[0]
    return np.squeeze(out).astype(np.float32)


def face_basis(forward: np.ndarray):
    up_hint = np.array([0, 1.0, 0])
    if abs(np.dot(forward, up_hint)) > 0.9:
        up_hint = np.array([0, 0, 1.0])
    right = np.cross(up_hint, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    return right, up


def bilinear(img: np.ndarray, col: np.ndarray, row: np.ndarray) -> np.ndarray:
    h, w = img.shape
    c0 = np.clip(np.floor(col).astype(np.int32), 0, w - 2)
    r0 = np.clip(np.floor(row).astype(np.int32), 0, h - 2)
    fc = np.clip(col - c0, 0, 1)
    fr = np.clip(row - r0, 0, 1)
    return (img[r0, c0] * (1 - fc) * (1 - fr) + img[r0, c0 + 1] * fc * (1 - fr)
            + img[r0 + 1, c0] * (1 - fc) * fr + img[r0 + 1, c0 + 1] * fc * fr)


def ground_disparity(rays: np.ndarray, origin, terrain) -> np.ndarray:
    """Metric disparity where each downward ray actually meets the ground.

    Marches the ray against the swissALTI3D height field rather than against an
    assumed flat plane. Returns 0 where the ray never lands, so those samples
    drop out of the fit.
    """
    ox, oy, oz = origin
    out = np.zeros(rays.shape[0], np.float64)
    steps = np.concatenate([np.arange(1.0, 20.0, 0.5), np.arange(20.0, 160.0, 2.0)])
    prev_gap = None
    prev_t = None
    for tt in steps:
        px = ox + rays[:, 0] * tt
        py = oy + rays[:, 1] * tt
        pz = oz + rays[:, 2] * tt
        gh = np.array([terrain_at(terrain, float(a), float(b)) for a, b in zip(px, pz)])
        gap = py - gh                      # positive while still above ground
        if prev_gap is not None:
            crossed = (prev_gap > 0) & (gap <= 0) & (out == 0)
            if crossed.any():
                # Linear interpolation between the bracketing steps.
                f = prev_gap[crossed] / np.maximum(1e-6, prev_gap[crossed] - gap[crossed])
                out[crossed] = prev_t + f * (tt - prev_t)
        prev_gap, prev_t = gap, tt
    hit = out > 0.5
    disp = np.zeros(rays.shape[0], np.float64)
    disp[hit] = 1.0 / out[hit]
    return disp


def panorama_depth(rgb: np.ndarray, origin=None, terrain=None):
    """Metric depth per pixel, by solving all faces together.

    Depth Anything normalises every image it is given, so the five faces come
    back in five incompatible scales — measured disparity maxima on one panorama
    were 9.3, 10.7, 15.3, 8.5 and 5.6, all bottoming out at zero. Assembling them
    and fitting a single scale and shift to the result is meaningless, and it
    collapses every pixel to roughly the same distance.

    So solve for a scale and shift *per face*, from two kinds of equation:

      overlap  where two faces see the same direction, their metric disparities
               must agree — this ties the ring of faces into one scale
      ground   where a ray points below the horizon it eventually meets the
               ground, and swissALTI3D says exactly where. Assuming instead a
               flat level road is wrong in a way that matters: Zurich is hilly,
               and on a rising street a near-horizon ray meets tarmac far sooner
               than flat geometry predicts, so every far anchor is too distant
               and the whole fit compresses.

    The overlaps make the faces mutually consistent; the ground makes them
    metric. Neither alone is enough.
    """
    h, w = rgb.shape[:2]
    lon = (np.arange(w) + 0.5) / w * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(h) + 0.5) / h * np.pi
    LON, LAT = np.meshgrid(lon, lat)
    dirs = np.stack([np.cos(LAT) * np.sin(LON),
                     -np.sin(LAT),
                     np.cos(LAT) * np.cos(LON)], axis=2)

    half = np.tan(np.radians(FACE_FOV) / 2)
    per_face, inside_face, central = [], [], []

    for _, forward in FACES:
        fdirs = face_directions(forward, FACE)
        est = estimate(sample_equirect(rgb, fdirs))
        if est.shape != (FACE, FACE):
            est = np.asarray(Image.fromarray(est).resize((FACE, FACE), Image.BILINEAR))

        right, up = face_basis(forward)
        ahead = dirs @ forward
        with np.errstate(divide="ignore", invalid="ignore"):
            fu = (dirs @ right) / ahead
            fv = (dirs @ up) / ahead
        inside = (ahead > 1e-6) & (np.abs(fu) <= half) & (np.abs(fv) <= half)

        d = np.zeros((h, w), np.float32)
        if inside.any():
            col = (fu[inside] / half + 1) / 2 * FACE - 0.5
            row = (fv[inside] / half + 1) / 2 * FACE - 0.5
            d[inside] = bilinear(est, col, row)
        per_face.append(d)
        inside_face.append(inside)
        # How square-on the ray is: a face's own centre is far more trustworthy
        # than its corners, both for blending and for weighting the solve.
        c = np.zeros((h, w), np.float32)
        c[inside] = np.clip(ahead[inside], 0, 1) ** 2
        central.append(c)

    n = len(FACES)
    down = -dirs[..., 1]
    sub = (slice(None, None, 6), slice(None, None, 6))     # subsample for the solve

    rows, rhs, weights = [], [], []

    # --- overlap equations ---
    for i in range(n):
        for j in range(i + 1, n):
            both = inside_face[i][sub] & inside_face[j][sub]
            if both.sum() < 200:
                continue
            di = per_face[i][sub][both]
            dj = per_face[j][sub][both]
            wgt = np.minimum(central[i][sub][both], central[j][sub][both])
            keep = wgt > 0.15
            if keep.sum() < 200:
                continue
            di, dj, wgt = di[keep], dj[keep], wgt[keep]
            # Cap how many equations any one pair contributes, so a large
            # overlap cannot outvote the ground anchor.
            if di.size > 4000:
                pick = np.random.default_rng(0).choice(di.size, 4000, replace=False)
                di, dj, wgt = di[pick], dj[pick], wgt[pick]
            r = np.zeros((di.size, 2 * n), np.float64)
            r[:, 2 * i] = di
            r[:, 2 * i + 1] = 1
            r[:, 2 * j] = -dj
            r[:, 2 * j + 1] = -1
            rows.append(r)
            rhs.append(np.zeros(di.size))
            weights.append(wgt)

    # --- ground equations ---
    ground_total = 0
    for i in range(n):
        m = inside_face[i][sub] & (down[sub] > 0.030) & (down[sub] < 0.75)
        if m.sum() < 200:
            continue
        di = per_face[i][sub][m]
        if origin is None or terrain is None:
            true_disp = down[sub][m] / CAMERA_HEIGHT
        else:
            true_disp = ground_disparity(dirs[sub][m], origin, terrain)
        wgt = central[i][sub][m]
        keep = (wgt > 0.15) & (di > 1e-4) & (true_disp > 1e-6)
        if keep.sum() < 200:
            continue
        di, true_disp, wgt = di[keep], true_disp[keep], wgt[keep]

        # Balance the band in log distance, not in pixels. Road 3 m away fills a
        # huge wedge of the image; road 70 m away is a thin sliver near the
        # horizon — yet the sliver carries all the far-field information. Sampled
        # by pixel the fit is swamped by near ground, and the shift term rises
        # until 1/shift caps the whole scene at about 11 m, which is exactly the
        # ceiling this produced before.
        dist = 1.0 / np.maximum(true_disp, 1e-6)
        bins = np.clip(((np.log(dist) - np.log(2.5)) / (np.log(90.0) - np.log(2.5))
                        * 12).astype(int), 0, 11)
        rng = np.random.default_rng(1)
        chosen = []
        for bi in range(12):
            idx = np.flatnonzero(bins == bi)
            if idx.size == 0:
                continue
            take = min(idx.size, 500)
            chosen.append(rng.choice(idx, take, replace=False))
        if not chosen:
            continue
        sel = np.concatenate(chosen)
        di, true_disp, wgt = di[sel], true_disp[sel], wgt[sel]
        r = np.zeros((di.size, 2 * n), np.float64)
        r[:, 2 * i] = di
        r[:, 2 * i + 1] = 1
        rows.append(r)
        rhs.append(true_disp)
        # The ground is the only metric information in the whole system.
        weights.append(wgt * 3.0)
        ground_total += di.size

    if not rows or ground_total < 500:
        return None
    A = np.vstack(rows)
    b = np.concatenate(rhs)
    W = np.concatenate(weights)

    # Robust: parked cars, kerbs and anything standing on the road break the
    # flat-ground assumption, and a plain fit lets them drag the scale.
    sol = None
    for _ in range(4):
        Aw = A * W[:, None]
        sol, *_ = np.linalg.lstsq(Aw, b * W, rcond=None)
        resid = np.abs(A @ sol - b)
        scale = max(1e-6, 2.5 * np.median(resid))
        W = np.concatenate(weights) * np.clip(1.0 - resid / (4 * scale), 0.05, 1.0)

    # --- compose ---
    disp = np.zeros((h, w), np.float64)
    wsum = np.zeros((h, w), np.float64)
    for i in range(n):
        m = inside_face[i]
        a_i, b_i = sol[2 * i], sol[2 * i + 1]
        contrib = a_i * per_face[i][m] + b_i
        disp[m] += contrib * central[i][m]
        wsum[m] += central[i][m]
    good = wsum > 1e-6
    disp[good] /= wsum[good]

    solid = good & (disp > 1.0 / 300.0)
    depth = np.full((h, w), np.inf, np.float32)
    depth[solid] = (1.0 / disp[solid]).astype(np.float32)
    return np.clip(depth, 0.5, 300.0), solid, dirs, sol


def run_one() -> int:
    index = json.loads((DATA / "panoramas.json").read_text())
    pano = sorted(index["panoramas"], key=lambda p: p["index"])[len(index["panoramas"]) // 2]
    src = PANOS / pano["file"]
    rgb = np.asarray(Image.open(src).convert("RGB"))
    print(f"  {src.name}  {rgb.shape[1]}x{rgb.shape[0]}")

    world = json.loads((DATA / "zurich_world.json").read_text())
    terrain = world["terrain"]
    origin = (pano["pos"][0], pano["pos"][1], pano["pos"][2])
    fitted = panorama_depth(rgb, origin=origin, terrain=terrain)
    if fitted is None:
        print("  FAIL — not enough ground to anchor the scale", file=sys.stderr)
        return 1
    depth, good, dirs, sol = fitted
    print("  per-face scale/shift:")
    for k, (name, _) in enumerate(FACES):
        print(f"    {name:6s} scale {sol[2*k]:8.4f}  shift {sol[2*k+1]:+8.4f}")

    band = depth[(depth < 250) & good]
    print(f"  depth {np.percentile(band,2):.1f}..{np.percentile(band,98):.1f} m "
          f"(median {np.median(band):.1f} m)")

    # Sanity: the road straight ahead should come out at a plausible distance.
    h, w = depth.shape
    ahead = depth[int(h * 0.62), int(w * 0.5)]
    print(f"  road ~20° below horizon, straight ahead: {ahead:.1f} m "
          f"(geometry says {CAMERA_HEIGHT/np.sin(np.radians(20)):.1f} m)")

    vis = np.clip(depth, 0, 60) / 60
    vis = (1 - vis) ** 0.7
    out = HERE.parent / "images" / "pano_depth_preview.jpg"
    strip = np.concatenate([
        np.asarray(Image.fromarray(rgb).resize((w // 2, h // 2))),
        np.asarray(Image.fromarray((vis * 255).astype(np.uint8)).convert("RGB")
                   .resize((w // 2, h // 2))),
    ], axis=0)
    Image.fromarray(strip).save(out, quality=90)
    print(f"  wrote {out}  (panorama above, depth below — nearer is brighter)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", action="store_true")
    args = ap.parse_args()
    raise SystemExit(run_one() if args.one else (ap.print_help() or 0))
