#!/usr/bin/env python3
"""
Remove cars, bikes and people from street imagery by temporal median compositing.

The trick needs no machine learning at all. A street gets driven several times on
different days, so the building stands still and everything else does not. Take
the per-pixel median across N rectified views of the same façade and anything
transient is outvoted by the building behind it.

Median rather than mean matters: a mean smears every car into a grey ghost,
whereas a median discards them outright as long as no single pixel is occluded in
more than half the views.

What survives is whatever was parked in the same spot every pass. That residue
is the (much smaller) job for segmentation and inpainting.

    python3 clean_plate.py --selftest    verify on synthetic captures
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import warnings

import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).parent


def median_composite(images: list[np.ndarray],
                     masks: list[np.ndarray] | None = None,
                     return_coverage: bool = False):
    """Per-pixel median across aligned views.

    `masks` optionally marks known-bad pixels (segmented vehicles, or areas the
    rectification could not fill). Masked pixels are excluded from the vote
    instead of being counted as evidence.
    """
    if not images:
        raise ValueError("no images")
    stack = np.stack(images).astype(np.float32)

    if masks is None:
        out = np.median(stack, axis=0).astype(np.uint8)
        if return_coverage:
            return out, np.ones(out.shape[:2], dtype=bool)
        return out

    valid = np.stack(masks).astype(bool)
    out = np.zeros(stack.shape[1:], dtype=np.float32)
    # Where every view is masked there is nothing to vote on; fall back to the
    # plain median so the result is never a hole.
    any_valid = valid.any(axis=0)
    filled = np.where(valid[..., None], stack, np.nan)
    # A pixel masked in every view produces an all-NaN slice; nanmedian is right
    # to complain, and `any_valid` below is what actually handles it.
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(filled, axis=0)
    plain = np.median(stack, axis=0)
    out = np.where(any_valid[..., None], np.nan_to_num(med, nan=0.0), plain)
    out = np.clip(out, 0, 255).astype(np.uint8)
    # Where no view had a usable pixel there is nothing to composite. Falling
    # back to the plain median there is actively wrong: it restores the very
    # occluder the masks excluded, which is how a permanently parked car
    # reappeared in the finished plate. Report it as a hole for inpainting.
    if return_coverage:
        return out, any_valid
    return out


def occlusion_report(images: list[np.ndarray], result: np.ndarray) -> dict:
    """How much each view disagreed with the composite — a proxy for how much of
    it was occluded, and a cheap way to spot a bad rectification."""
    res = result.astype(np.float32)
    out = []
    for i, img in enumerate(images):
        diff = np.abs(img.astype(np.float32) - res).mean(axis=2)
        out.append({"view": i, "occluded_fraction": float((diff > 40).mean())})
    return {"views": out}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def synthetic_facade(size=(768, 512)) -> np.ndarray:
    """A stand-in façade, built from a real scanned plaster material if one has
    been fetched, so the test exercises real texture statistics rather than flat
    colour."""
    w, h = size
    plaster = HERE.parent / "data" / "materials" / "wall_albedo.jpg"
    if plaster.exists():
        img = Image.open(plaster).convert("RGB")
        img = img.crop((0, 0, img.width, img.width)).resize((w, h), Image.LANCZOS)
        base = np.array(img).astype(np.float32)
    else:
        base = np.full((h, w, 3), 190, np.float32)
        base += np.random.default_rng(0).normal(0, 6, base.shape)

    # Windows, so there is real structure to preserve.
    for row in range(1, 4):
        for col in range(6):
            x0 = 40 + col * 118
            y0 = 40 + row * 118
            base[y0:y0 + 78, x0:x0 + 74] = (38, 44, 56)
            base[y0 - 5:y0, x0 - 5:x0 + 79] = (225, 222, 214)
    return np.clip(base, 0, 255).astype(np.uint8)


def add_clutter(plate: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Park some cars and a cyclist in front, in different places each pass."""
    img = plate.copy()
    h, w, _ = img.shape
    for _ in range(rng.integers(2, 4)):        # cars along the kerb
        cw = int(rng.integers(150, 230))
        ch = int(rng.integers(70, 100))
        x = int(rng.integers(0, max(1, w - cw)))
        y = h - ch - int(rng.integers(0, 30))
        colour = rng.integers(20, 235, 3)
        img[y:y + ch, x:x + cw] = colour
        img[y + ch // 3: y + ch // 2, x + 10:x + cw - 10] = np.clip(colour + 40, 0, 255)
    for _ in range(rng.integers(1, 3)):        # cyclists / pedestrians
        bw, bh = int(rng.integers(30, 55)), int(rng.integers(90, 140))
        x = int(rng.integers(0, max(1, w - bw)))
        y = h - bh
        img[y:y + bh, x:x + bw] = rng.integers(10, 90, 3)
    return img


def selftest(views: int = 12) -> int:
    rng = np.random.default_rng(7)
    truth = synthetic_facade()

    # Keep the clutter masks the synthesis produced: in the real pipeline these
    # come from semantic segmentation, imperfectly. Here they are exact, so the
    # masked figure is an upper bound on what segmentation can buy.
    captures, masks = [], []
    for _ in range(views):
        img = add_clutter(truth, rng)
        occluded = (np.abs(img.astype(np.int16) - truth.astype(np.int16)).sum(axis=2) > 12)
        captures.append(img)
        masks.append(~occluded)          # True = usable pixel

    plain = median_composite(captures)
    masked = median_composite(captures, masks)

    # Score only where anything was ever occluded. Averaged over the whole
    # façade, the untouched upper storeys drown out the band that matters.
    contested = np.stack([~m for m in masks]).any(axis=0)
    def err(img):
        d = np.abs(img.astype(np.float32) - truth.astype(np.float32)).mean(axis=2)
        return float(d[contested].mean())

    single = min(err(c) for c in captures)
    coverage = np.stack([~m for m in masks]).mean(axis=0)[contested].mean()

    strip = Image.new("RGB", (truth.shape[1] * 4, truth.shape[0]))
    for i, img in enumerate([captures[0], plain, masked, truth]):
        strip.paste(Image.fromarray(img), (truth.shape[1] * i, 0))
    out = HERE.parent / "images" / "clean_plate_selftest.jpg"
    out.parent.mkdir(exist_ok=True)
    strip.save(out, quality=92)

    print(f"  {views} synthetic passes, cars and cyclists reshuffled each time")
    print(f"  contested area is occluded in {coverage*100:.0f}% of views on average")
    print(f"  error over contested pixels, 0-255:")
    print(f"    best single capture   {single:7.2f}")
    print(f"    median only           {err(plain):7.2f}")
    print(f"    median + masks        {err(masked):7.2f}")
    print(f"  wrote {out}  (capture | median | median+masks | truth)")

    if err(masked) < single / 4 and err(plain) < single:
        print("  PASS — transients removed; masks clear the heavily-occluded band")
        return 0
    print("  FAIL — compositing did not clean the contested band", file=sys.stderr)
    return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--views", type=int, default=12)
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(selftest(args.views))
    ap.print_help()
