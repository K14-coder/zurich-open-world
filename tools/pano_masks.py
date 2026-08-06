#!/usr/bin/env python3
"""
Segment each panorama so the photograph can carve its own silhouette.

Building geometry is extruded from OSM footprints with an estimated height, so
our boxes are routinely taller and wider than the buildings actually are. The
projection does not know that: it happily paints whatever the photograph holds
in that direction onto the surplus box, which is why hard rectangles of sky,
trees and distant buildings float where a roofline should be.

The photograph knows, though. Where it shows sky there is nothing solid, so the
box should not be drawn there at all — the image becomes a silhouette mask and
the surplus geometry is carved away.

Writes one grayscale mask per panorama, which the renderer packs into the alpha
channel of the texture it already allocates.

    0   sky        — nothing solid in this direction
    85  ground     — road, sidewalk, terrain
    170 structure  — building, wall, fence
    255 other      — vegetation, poles, vehicles, people
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
from PIL import Image

import segment

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent / "data"
PANOS = DATA / "panoramas"

SKY, GROUND, STRUCTURE, OTHER = 0, 85, 170, 255

GROUND_CLASSES = {segment.ROAD, segment.SIDEWALK, segment.TERRAIN}
STRUCTURE_CLASSES = {segment.BUILDING, segment.WALL, segment.FENCE}


def categorise(ids: np.ndarray) -> np.ndarray:
    out = np.full(ids.shape, OTHER, dtype=np.uint8)
    out[np.isin(ids, list(GROUND_CLASSES))] = GROUND
    out[np.isin(ids, list(STRUCTURE_CLASSES))] = STRUCTURE
    out[ids == segment.SKY] = SKY
    return out


def build() -> None:
    index = json.loads((DATA / "panoramas.json").read_text())
    panos = index["panoramas"]
    print(f"  segmenting {len(panos)} panoramas...")

    stats = {"sky": 0.0, "structure": 0.0, "ground": 0.0}
    for i, pano in enumerate(panos):
        src = PANOS / pano["file"]
        dst = src.with_name(src.stem + "_mask.png")
        if dst.exists():
            continue
        rgb = np.asarray(Image.open(src).convert("RGB"))

        # An equirectangular frame is 2:1, and squashing it into the model's
        # square input distorts every vertical. Segmenting the two halves
        # separately keeps each one close to square, which matters most exactly
        # where it matters here: the roofline against the sky.
        h, w = rgb.shape[:2]
        halves = [rgb[:, : w // 2], rgb[:, w // 2 :]]
        cats = [categorise(segment.labels(half)) for half in halves]
        mask = np.concatenate(cats, axis=1)

        Image.fromarray(mask).save(dst)
        stats["sky"] += float((mask == SKY).mean())
        stats["structure"] += float((mask == STRUCTURE).mean())
        stats["ground"] += float((mask == GROUND).mean())
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(panos)}", flush=True)

    n = max(1, len(panos))
    print(f"  average composition: sky {stats['sky']/n*100:.0f}%, "
          f"structure {stats['structure']/n*100:.0f}%, "
          f"ground {stats['ground']/n*100:.0f}%")
    print(f"  wrote masks alongside {PANOS}")


if __name__ == "__main__":
    build()
