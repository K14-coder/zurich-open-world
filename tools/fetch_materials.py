#!/usr/bin/env python3
"""
Download scanned PBR materials for the façades and the carriageway.

Everything on screen up to now was invented by arithmetic — window rhythm from a
hash, asphalt from a grain function. That yields a plausible city but never a
real-looking one, because a hash has no idea what plaster looks like. These are
photographed materials with real albedo, normal and roughness maps.

Source is ambientCG, which publishes under CC0: free, commercial use allowed,
no attribution required. Materials are packed into layered arrays the renderer
binds as texture arrays, one layer per material.
"""

from __future__ import annotations

import io
import json
import pathlib
import sys
import urllib.request
import zipfile

from PIL import Image

HERE = pathlib.Path(__file__).parent
OUT = HERE.parent / "data" / "materials"
CACHE = HERE / "cache" / "materials"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

SIZE = 1024   # per-layer resolution

# Zurich's centre is render and painted plaster over stone, with concrete for
# the post-war blocks. Deliberately no brick: it is rare here and reads as
# northern-German the moment it appears.
WALLS = [
    "Plaster001", "Plaster002", "Plaster003", "Plaster004",
    "PaintedPlaster017", "Concrete034", "Concrete036", "Concrete030",
]
ROADS = ["Asphalt031"]

MAPS = {"Color": "albedo", "NormalGL": "normal", "Roughness": "roughness"}


def download(asset: str) -> pathlib.Path:
    path = CACHE / f"{asset}_1K-JPG.zip"
    if path.exists():
        return path
    url = f"https://ambientcg.com/get?file={asset}_1K-JPG.zip"
    print(f"  downloading {asset}...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "zurich-open-world/1.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        path.write_bytes(resp.read())
    return path


def extract(asset: str) -> dict[str, Image.Image]:
    path = download(asset)
    found: dict[str, Image.Image] = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            for suffix, kind in MAPS.items():
                if name.endswith(f"_{suffix}.jpg") or name.endswith(f"_{suffix}.png"):
                    img = Image.open(io.BytesIO(z.read(name)))
                    found[kind] = img.convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
    return found


def pack(assets: list[str], prefix: str) -> None:
    layers: dict[str, list[Image.Image]] = {k: [] for k in MAPS.values()}
    used = []
    for asset in assets:
        try:
            maps = extract(asset)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {asset}: {exc}", file=sys.stderr)
            continue
        if "albedo" not in maps:
            print(f"  ! {asset}: no colour map", file=sys.stderr)
            continue
        # Roughness ships as greyscale; a missing normal is flat blue.
        maps.setdefault("normal", Image.new("RGB", (SIZE, SIZE), (128, 128, 255)))
        maps.setdefault("roughness", Image.new("RGB", (SIZE, SIZE), (180, 180, 180)))
        for kind in MAPS.values():
            layers[kind].append(maps[kind])
        used.append(asset)

    if not used:
        raise SystemExit(f"no {prefix} materials could be fetched")

    # One tall strip per map: layer N occupies rows [N*SIZE, (N+1)*SIZE). The
    # renderer slices it into a texture array at load.
    for kind, images in layers.items():
        strip = Image.new("RGB", (SIZE, SIZE * len(images)))
        for i, img in enumerate(images):
            strip.paste(img, (0, i * SIZE))
        path = OUT / f"{prefix}_{kind}.jpg"
        strip.save(path, "JPEG", quality=90, optimize=True)
        print(f"  {path.name}: {len(images)} layers, {path.stat().st_size/1e6:.1f} MB")

    (OUT / f"{prefix}.json").write_text(json.dumps(
        {"size": SIZE, "layers": len(used), "assets": used,
         "licence": "CC0 via ambientCG"}, indent=2))


if __name__ == "__main__":
    print("Walls:")
    pack(WALLS, "wall")
    print("Roads:")
    pack(ROADS, "road")
    print("done")
