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

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent / "data"
PANOS = DATA / "panoramas"
MODEL = pathlib.Path.home() / ".cache" / "zurich-models" / "depth_anything_v2_small.onnx"
MODEL_URL = ("https://huggingface.co/onnx-community/depth-anything-v2-small/"
             "resolve/main/onnx/model.onnx")

FACE = 518          # Depth Anything V2 input size
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

    t = (np.arange(size) + 0.5) / size * 2 - 1      # -1..1 across a 90° face
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


def panorama_depth(rgb: np.ndarray):
    """Metric-shaped inverse depth per pixel of an equirectangular panorama.

    Gathers rather than scatters. Pushing 518x518 samples per face outward into
    a 3072x1536 grid leaves three quarters of it empty, and no amount of hole
    filling recovers that — the first version of this produced a uniform depth
    map because the scale fit saw almost nothing but road. Asking each output
    pixel which face it came from cannot leave a hole.
    """
    h, w = rgb.shape[:2]
    lon = (np.arange(w) + 0.5) / w * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(h) + 0.5) / h * np.pi
    LON, LAT = np.meshgrid(lon, lat)
    dirs = np.stack([np.cos(LAT) * np.sin(LON),
                     -np.sin(LAT),
                     np.cos(LAT) * np.cos(LON)], axis=2)

    inv_depth = np.zeros((h, w), np.float32)
    valid = np.zeros((h, w), bool)

    for _, forward in FACES:
        fdirs = face_directions(forward, FACE)
        face = sample_equirect(rgb, fdirs)
        est = estimate(face)
        if est.shape != (FACE, FACE):
            est = np.asarray(Image.fromarray(est).resize((FACE, FACE), Image.BILINEAR))

        right, up = face_basis(forward)
        ahead = dirs @ forward
        u = (dirs @ right)
        v = (dirs @ up)
        with np.errstate(divide="ignore", invalid="ignore"):
            fu = u / ahead
            fv = v / ahead
        # Inside this face, with a sliver of margin so neighbours meet cleanly.
        inside = (ahead > 1e-6) & (np.abs(fu) <= 1.001) & (np.abs(fv) <= 1.001)
        if not inside.any():
            continue
        col = (fu[inside] + 1) / 2 * FACE - 0.5
        row = (fv[inside] + 1) / 2 * FACE - 0.5
        inv_depth[inside] = bilinear(est, col, row)
        valid[inside] = True

    return inv_depth, valid, dirs


def anchor_to_metres(inv_depth, valid, dirs):
    """Fit scale and shift so the road comes out at its true distance.

    Depth Anything returns relative inverse depth: correct in shape, arbitrary
    in units. The ground gives the anchor, because for any ray pointing below
    the horizon the distance to a flat road is camera height over the sine of
    the angle below it. Fitting against those rays converts the whole map to
    metres without needing any other measurement.
    """
    down = -dirs[..., 1]
    # The band has to be wide, and that is the whole difficulty. Rays steeply
    # below the horizon hit the road 3 m away; rays just below it hit the same
    # road 70 m away. Fitting only the steep ones gives an 8 m baseline, and
    # extrapolating a disparity fit from 8 m out to 100 m collapses the slope to
    # nothing — every pixel then lands at the same distance, which is exactly
    # what the first attempt produced. Below 0.75 excludes the capture vehicle.
    band = (down > 0.030) & (down < 0.75) & valid
    if band.sum() < 500:
        return None
    true_depth = CAMERA_HEIGHT / down[band]
    inv_true = 1.0 / true_depth
    x = inv_depth[band]
    # Sky leaks into the shallow end of the band on an uphill street; its
    # disparity is near zero and it would drag the fit.
    alive = x > np.percentile(inv_depth[valid], 2)
    x, inv_true = x[alive], inv_true[alive]
    if x.size < 500:
        return None

    # Robust least squares: a fifth of that band is the car's bonnet, kerbs and
    # painted lines rather than clean road, and those would drag a plain fit.
    keep = np.ones(x.shape, bool)
    scale = shift = 0.0
    for _ in range(4):
        A = np.stack([x[keep], np.ones(keep.sum())], axis=1)
        sol, *_ = np.linalg.lstsq(A, inv_true[keep], rcond=None)
        scale, shift = float(sol[0]), float(sol[1])
        resid = np.abs(scale * x + shift - inv_true)
        keep = resid < max(1e-4, 2.5 * np.median(resid))
        if keep.sum() < 200:
            break

    metric_inv = scale * inv_depth + shift
    depth = np.full(inv_depth.shape, np.inf, np.float32)
    good = metric_inv > 1e-3
    depth[good] = 1.0 / metric_inv[good]
    return np.clip(depth, 0.5, 300.0), good


def run_one() -> int:
    index = json.loads((DATA / "panoramas.json").read_text())
    pano = sorted(index["panoramas"], key=lambda p: p["index"])[len(index["panoramas"]) // 2]
    src = PANOS / pano["file"]
    rgb = np.asarray(Image.open(src).convert("RGB"))
    print(f"  {src.name}  {rgb.shape[1]}x{rgb.shape[0]}")

    inv, valid, dirs = panorama_depth(rgb)
    fitted = anchor_to_metres(inv, valid, dirs)
    if fitted is None:
        print("  FAIL — not enough ground to anchor the scale", file=sys.stderr)
        return 1
    depth, good = fitted

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
