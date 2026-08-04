#!/usr/bin/env python3
"""
Per-pixel semantic segmentation of street photographs, to decide which pixels
are actually façade.

The clean-plate median needs to *exclude* occluded pixels rather than let them
vote. With only three or four usable views per wall — which is what the quality
gates leave — a parked car present in two of them survives a plain median. The
synthetic test measured this precisely: masks took the error from 20.1 to 0.00.

Model is SegFormer-B0 fine-tuned on Cityscapes, run locally through
onnxruntime. Cityscapes is the right training set by a wide margin: it is street
scenes photographed from a car, which is exactly what Mapillary is.

The mask is built *positively* — keep `building` and `wall` — rather than by
listing things to remove. Enumerating occluders always misses one (scaffolding,
awnings, wheelie bins, a delivery van's roof box); asking "is this façade?"
cannot.

    python3 segment.py --selftest    run on a cached Mapillary frame
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = pathlib.Path(__file__).parent
MODEL = pathlib.Path.home() / ".cache" / "zurich-models" / "segformer_cityscapes.onnx"
MASK_CACHE = HERE / "cache" / "masks"
MASK_CACHE.mkdir(parents=True, exist_ok=True)

# Cityscapes train ids.
ROAD, SIDEWALK, BUILDING, WALL, FENCE, POLE = 0, 1, 2, 3, 4, 5
TRAFFIC_LIGHT, TRAFFIC_SIGN, VEGETATION, TERRAIN, SKY = 6, 7, 8, 9, 10
PERSON, RIDER, CAR, TRUCK, BUS, TRAIN, MOTORCYCLE, BICYCLE = 11, 12, 13, 14, 15, 16, 17, 18

CLASS_NAMES = {
    ROAD: "road", SIDEWALK: "sidewalk", BUILDING: "building", WALL: "wall",
    FENCE: "fence", POLE: "pole", TRAFFIC_LIGHT: "traffic light",
    TRAFFIC_SIGN: "traffic sign", VEGETATION: "vegetation", TERRAIN: "terrain",
    SKY: "sky", PERSON: "person", RIDER: "rider", CAR: "car", TRUCK: "truck",
    BUS: "bus", TRAIN: "train", MOTORCYCLE: "motorcycle", BICYCLE: "bicycle",
}
FACADE_CLASSES = {BUILDING, WALL}

# SegFormer expects ImageNet normalisation.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT = 1024

_session: ort.InferenceSession | None = None


MODEL_URL = ("https://huggingface.co/Xenova/segformer-b0-finetuned-cityscapes-"
             "1024-1024/resolve/main/onnx/model.onnx")


def ensure_model() -> None:
    """Fetch the 15 MB model on first use so a fresh clone just works."""
    if MODEL.exists():
        return
    import urllib.request
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching segmentation model -> {MODEL}", flush=True)
    tmp = MODEL.with_suffix(".part")
    with urllib.request.urlopen(MODEL_URL, timeout=600) as r:
        tmp.write_bytes(r.read())
    tmp.rename(MODEL)


def session() -> ort.InferenceSession:
    global _session
    if _session is None:
        ensure_model()
        if not MODEL.exists():
            raise SystemExit(f"model missing: {MODEL}")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _session = ort.InferenceSession(str(MODEL), opts,
                                        providers=["CPUExecutionProvider"])
    return _session


def labels(rgb: np.ndarray) -> np.ndarray:
    """Per-pixel Cityscapes class ids, at the input image's own resolution."""
    h, w = rgb.shape[:2]
    img = Image.fromarray(rgb.astype(np.uint8)).resize((INPUT, INPUT), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    x = np.transpose(x, (2, 0, 1))[None]

    out = session().run(None, {"pixel_values": x})[0][0]      # (C, h', w')
    ids = np.argmax(out, axis=0).astype(np.uint8)
    # SegFormer emits logits at a quarter resolution; nearest-neighbour back up
    # keeps class boundaries crisp instead of inventing intermediate ids.
    return np.asarray(
        Image.fromarray(ids).resize((w, h), Image.NEAREST))


def facade_mask(rgb: np.ndarray, cache_key: str | None = None) -> np.ndarray:
    """True where the pixel is building façade and can be trusted."""
    if cache_key:
        path = MASK_CACHE / f"{cache_key}.png"
        if path.exists():
            return np.asarray(Image.open(path)) > 127
    ids = labels(rgb)
    mask = np.isin(ids, list(FACADE_CLASSES))
    if cache_key:
        Image.fromarray((mask * 255).astype(np.uint8)).save(MASK_CACHE / f"{cache_key}.png")
    return mask


def composition(ids: np.ndarray) -> list[tuple[str, float]]:
    total = ids.size
    out = []
    for cid, name in CLASS_NAMES.items():
        frac = float((ids == cid).mean())
        if frac > 0.005:
            out.append((name, frac))
    return sorted(out, key=lambda r: -r[1])


def selftest() -> int:
    frames = sorted((HERE / "cache" / "mly_images").glob("*.jpg"))
    if not frames:
        raise SystemExit("no cached Mapillary frames; run rectify.py first")

    # Pick a frame with plenty of content rather than the first alphabetically.
    frame = max(frames[:60], key=lambda p: p.stat().st_size)
    rgb = np.asarray(Image.open(frame).convert("RGB"))
    print(f"  {frame.name}  {rgb.shape[1]}x{rgb.shape[0]}")

    ids = labels(rgb)
    for name, frac in composition(ids)[:9]:
        print(f"    {name:14s} {frac*100:5.1f}%")

    mask = np.isin(ids, list(FACADE_CLASSES))
    occluders = np.isin(ids, [PERSON, RIDER, CAR, TRUCK, BUS, MOTORCYCLE, BICYCLE])
    print(f"  façade {mask.mean()*100:.1f}% of frame, "
          f"vehicles and people {occluders.mean()*100:.1f}%")

    # Overlay: façade kept in colour, everything else dimmed, occluders in red.
    over = rgb.astype(np.float32) * 0.25
    over[mask] = rgb[mask]
    tint = over.copy()
    tint[occluders] = np.array([220, 40, 40], dtype=np.float32)
    strip = np.concatenate([rgb, tint.astype(np.uint8)], axis=1)
    out = HERE.parent / "images" / "segmentation_selftest.jpg"
    Image.fromarray(strip).save(out, quality=90)
    print(f"  wrote {out}  (frame | façade kept, vehicles in red)")

    if mask.mean() < 0.05:
        print("  FAIL — almost nothing classified as façade", file=sys.stderr)
        return 1
    print("  PASS")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(selftest())
    ap.print_help()
