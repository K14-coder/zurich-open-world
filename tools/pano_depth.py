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
# Metric, not relative. The relative model needs its arbitrary units fitted to
# metres, and that fit is the whole difficulty: an affine transform per face,
# tied by overlaps and anchored on the ground. Six attempts at it moved the depth
# ceiling from 6 m to 18 m and no further, because the shift term caps every
# scene at 1/shift. A metric model states metres outright and the entire problem
# disappears.
MODEL = pathlib.Path.home() / ".cache" / "zurich-models" / "depth_metric.onnx"
MODEL_URL = ("https://huggingface.co/77ukhtar/depth-anything-v2-metric-onnx/"
             "resolve/main/model.onnx")

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
    """Metric depth in metres for one perspective face."""
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


def panorama_depth(rgb: np.ndarray, origin=None, terrain=None):
    """Metric depth per pixel of an equirectangular panorama.

    Every face comes back in metres already, so the faces are directly
    comparable and there is nothing to solve — no per-face scale, no shift, no
    ground anchor, no overlap equations. All of that machinery existed only to
    convert a relative model's arbitrary units, and it never worked.

    Faces still overlap at 110°, but now only so that neighbours can be blended
    by how square-on each ray is, rather than to tie their scales together.
    """
    h, w = rgb.shape[:2]
    lon = (np.arange(w) + 0.5) / w * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(h) + 0.5) / h * np.pi
    LON, LAT = np.meshgrid(lon, lat)
    dirs = np.stack([np.cos(LAT) * np.sin(LON),
                     -np.sin(LAT),
                     np.cos(LAT) * np.cos(LON)], axis=2)

    half = np.tan(np.radians(FACE_FOV) / 2)
    accum = np.zeros((h, w), np.float64)
    wsum = np.zeros((h, w), np.float64)

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
        if not inside.any():
            continue

        col = (fu[inside] / half + 1) / 2 * FACE - 0.5
        row = (fv[inside] / half + 1) / 2 * FACE - 0.5
        d = bilinear(est, col, row)

        # A perspective face reports depth along its own axis; the distance we
        # want is along the ray. Without this every face is short towards its
        # corners, by up to the 1/cos of the face half-angle.
        d = d / np.clip(ahead[inside], 1e-3, 1.0)

        weight = np.clip(ahead[inside], 0, 1) ** 2
        accum[inside] += d * weight
        wsum[inside] += weight

    solid = wsum > 1e-6
    depth = np.full((h, w), np.inf, np.float32)
    depth[solid] = (accum[solid] / wsum[solid]).astype(np.float32)
    return np.clip(depth, 0.5, 300.0), solid, dirs, None


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
    for name, fwd in FACES:
        d = estimate(sample_equirect(rgb, face_directions(fwd, FACE)))
        print(f"    {name:6s} {d.min():6.1f}..{d.max():6.1f} m  median {np.median(d):6.1f} m")

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


def run_all() -> int:
    """Depth for every panorama, written beside it as a 16-bit PNG in centimetres.

    16-bit because 8 would quantise a 60 m street into 24 cm steps, and that
    shows as terracing on any surface you drive past slowly.
    """
    index = json.loads((DATA / "panoramas.json").read_text())
    panos = sorted(index["panoramas"], key=lambda p: p["index"])
    world = json.loads((DATA / "zurich_world.json").read_text())
    made = 0
    for i, pano in enumerate(panos):
        src = PANOS / pano["file"]
        dst = src.with_name(src.stem + "_depth.png")
        if dst.exists():
            continue
        rgb = np.asarray(Image.open(src).convert("RGB"))
        fitted = panorama_depth(rgb, origin=tuple(pano["pos"]),
                                terrain=world["terrain"])
        if fitted is None:
            print(f"    ! {src.name}: no depth", file=sys.stderr)
            continue
        depth, solid, _, _ = fitted
        cm = np.zeros(depth.shape, np.uint16)
        cm[solid] = np.clip(depth[solid] * 100.0, 1, 65535).astype(np.uint16)
        Image.fromarray(cm).save(dst)
        made += 1
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(panos)}", flush=True)
    print(f"  wrote {made} depth maps")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all:
        raise SystemExit(run_all())
    raise SystemExit(run_one() if args.one else (ap.print_help() or 0))
