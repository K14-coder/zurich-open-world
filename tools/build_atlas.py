#!/usr/bin/env python3
"""
Pack the finished façade plates into one atlas the renderer can bind.

Each plate is hung on the plane the fit chose, not on the OSM wall, so the atlas
entry carries the four world-space corners alongside its UV rect. The renderer
then draws a textured quad in front of the extruded wall and needs to know
nothing about how the plate was made.
"""

from __future__ import annotations

import json
import pathlib

from PIL import Image

HERE = pathlib.Path(__file__).parent
PLATES = HERE.parent / "images" / "facades"
OUT = HERE.parent / "data"
ATLAS = 4096   # 2048 overflowed at 22 plates; leave headroom for the district
PAD = 2


import math
import numpy as np


def plate_fault(entry: dict, img: Image.Image) -> str | None:
    """Why this plate should not ship, or None if it is fit to use.

    A failed reconstruction is worse than no reconstruction: the procedural
    façade underneath is plausible, whereas a broken plate is a smear of sky and
    a distant building pasted onto a wall, and it reads as a bug rather than as
    a texture.
    """
    if entry.get("views", 0) < 4:
        return f"only {entry.get('views', 0)} views"
    if entry.get("holes", 0) > 0.15:
        return f"{entry['holes']*100:.0f}% inpainted"

    a, b = entry["corners"][0], entry["corners"][1]
    span = math.hypot(b[0] - a[0], b[2] - a[2])
    # A very long wall is never one flat façade in this part of Zurich; it is
    # several buildings the footprint merged, and no single plane fits it.
    if span > 22.0:
        return f"{span:.0f} m wide, too long to be one plane"

    px = np.asarray(img).astype(np.float32)
    blue = (px[..., 2] > px[..., 0] + 12) & (px[..., 2] > 90)
    if blue.mean() > 0.18:
        return f"{blue.mean()*100:.0f}% sky — plane or height is wrong"
    # A plate that is mostly one flat tone captured nothing useful.
    if px.reshape(-1, 3).std(axis=0).mean() < 18:
        return "no detail"
    return None


def build() -> None:
    index = json.loads((PLATES / "plates.json").read_text())["plates"]
    images, rejected = [], []
    for entry in index:
        path = PLATES / entry["file"]
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        why = plate_fault(entry, img)
        if why:
            rejected.append((entry["file"], why))
            continue
        images.append((entry, img))
    for name, why in rejected:
        print(f"  rejected {name}: {why}")
    if not images:
        raise SystemExit("no plates to pack")

    # Shelf packing, tallest first. Plates are all roughly portrait and similar
    # size, so anything cleverer would buy nothing.
    images.sort(key=lambda p: -p[1].height)
    atlas = Image.new("RGB", (ATLAS, ATLAS), (128, 128, 128))
    x = y = shelf = 0
    packed = []
    for entry, img in images:
        w, h = img.size
        if w + PAD > ATLAS:
            scale = (ATLAS - PAD) / w
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            w, h = img.size
        if x + w + PAD > ATLAS:
            x = 0
            y += shelf + PAD
            shelf = 0
        if y + h + PAD > ATLAS:
            print(f"  ! atlas full, dropped {entry['file']}")
            continue
        atlas.paste(img, (x, y))
        packed.append({
            "corners": entry["corners"],
            "uv": [round(x / ATLAS, 6), round(y / ATLAS, 6),
                   round((x + w) / ATLAS, 6), round((y + h) / ATLAS, 6)],
            "road": entry.get("road", ""),
        })
        x += w + PAD
        shelf = max(shelf, h)

    atlas.save(OUT / "facade_atlas.jpg", "JPEG", quality=94, optimize=True)
    (OUT / "facade_atlas.json").write_text(
        json.dumps({"size": ATLAS, "plates": packed}, separators=(",", ":")))
    used = y + shelf
    print(f"  packed {len(packed)}/{len(images)} plates into {ATLAS}x{ATLAS} "
          f"({100*used/ATLAS:.0f}% of height used)")
    print(f"  wrote {OUT/'facade_atlas.jpg'} "
          f"({(OUT/'facade_atlas.jpg').stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    build()
