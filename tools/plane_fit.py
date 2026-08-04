#!/usr/bin/env python3
"""
Solve for where a façade actually is, instead of trusting the OSM footprint.

Rectified views of the same wall refused to register, and patching that in 2D was
treating the symptom. The cause is that the plane is wrong: an OSM footprint sits
a metre or two off the true wall face, its orientation is a little out, and each
photograph is at a different distance — so every view lands differently and no
single 2D shift can reconcile them.

Fix the plane and the views register for free.

The objective is the sharpness of the median composite. Misaligned views blur
each other, aligned ones do not, so gradient energy in the composite peaks
exactly when the plane is right. It needs no ground truth and no feature
matching, which is what defeated the previous attempt — façades are made of
repeating window bays, and a matcher will happily align one bay to the next.
"""

from __future__ import annotations

import math

import numpy as np

# Search bounds. A footprint is wrong by metres, not tens of metres.
OFFSET_RANGE = 2.5        # metres along the wall normal
YAW_RANGE = math.radians(5)
# Only fit against the lower part of the wall: building heights are estimated,
# so the top is the least trustworthy part and often sky.
FIT_HEIGHT = 12.0
FIT_PPM = 8               # pixels per metre while searching (coarse is enough)


def plane_grid(f: dict, offset: float, yaw: float, base_dy: float,
               ppm: float, height: float):
    """World-space sample points for a candidate plane."""
    ax, az = f["a"]
    bx, bz = f["z"]
    nx, nz = f["n"]
    mx, mz = (ax + bx) / 2, (az + bz) / 2

    # Rotate the wall segment about its own midpoint, then push it along its
    # normal. Those two corrections cover essentially all footprint error.
    c, s = math.cos(yaw), math.sin(yaw)
    def rot(px, pz):
        dx, dz = px - mx, pz - mz
        return mx + dx * c - dz * s, mz + dx * s + dz * c
    ax2, az2 = rot(ax, az)
    bx2, bz2 = rot(bx, bz)
    nx2 = nx * c - nz * s
    nz2 = nx * s + nz * c
    ax2 += nx2 * offset; az2 += nz2 * offset
    bx2 += nx2 * offset; bz2 += nz2 * offset

    length = math.hypot(bx2 - ax2, bz2 - az2)
    W = max(16, int(length * ppm))
    H = max(16, int(height * ppm))
    u = (np.arange(W) + 0.5) / W
    v = (np.arange(H) + 0.5) / H
    U, V = np.meshgrid(u, v)

    Px = ax2 + (bx2 - ax2) * U
    Pz = az2 + (bz2 - az2) * U
    Py = f["base"] + base_dy + height * (1.0 - V)
    return Px, Py, Pz


def highpass(g: np.ndarray) -> np.ndarray:
    """Strip the low frequencies before correlating.

    Plain NCC has its own degeneracy, just as sharpness did. Sliding the plane
    towards the camera magnifies the sampling, which smooths every view, and
    smooth images correlate better — so the fit pinned at the near edge of its
    search on every façade. Correlating only the detail removes that reward: a
    magnified blur has no high frequencies to agree about.
    """
    import cv2
    return g - cv2.GaussianBlur(g, (0, 0), 2.0)


def quad_corners(f: dict, offset: float, yaw: float, base_dy: float,
                 height: float) -> list:
    """World-space corners of the fitted plane, for the renderer to hang the
    finished plate on. Returned bottom-left, bottom-right, top-left, top-right
    in the same parameterisation `plane_grid` samples."""
    ax, az = f["a"]
    bx, bz = f["z"]
    nx, nz = f["n"]
    mx, mz = (ax + bx) / 2, (az + bz) / 2
    c, s = math.cos(yaw), math.sin(yaw)

    def rot(px, pz):
        dx, dz = px - mx, pz - mz
        return mx + dx * c - dz * s, mz + dx * s + dz * c

    ax2, az2 = rot(ax, az)
    bx2, bz2 = rot(bx, bz)
    nx2 = nx * c - nz * s
    nz2 = nx * s + nz * c
    # Nudge towards the street so the plate sits in front of the extruded wall
    # rather than co-planar with it, which would z-fight.
    push = 0.06
    ax2 += nx2 * (offset + push); az2 += nz2 * (offset + push)
    bx2 += nx2 * (offset + push); bz2 += nz2 * (offset + push)

    y0 = f["base"] + base_dy
    y1 = y0 + height
    return [[round(ax2, 3), round(y0, 3), round(az2, 3)],
            [round(bx2, 3), round(y0, 3), round(bz2, 3)],
            [round(ax2, 3), round(y1, 3), round(az2, 3)],
            [round(bx2, 3), round(y1, 3), round(bz2, 3)]]


def agreement(views: list, masks: list, plate: np.ndarray,
              weights: list | None = None) -> float:
    """Mean normalised cross-correlation of each view against the composite.

    NOT gradient energy. Sharpness looks like the obvious objective — blur is
    what misalignment produces — but it rewards *compression*: a plane tilted
    nearly edge-on squeezes the façade into a narrow strip and packs enormous
    gradient into every pixel. The first version of this fitter duly ran to the
    edge of its search bounds and produced an oblique smear.

    NCC cannot be gamed that way. It is invariant to brightness and contrast
    scaling, so it measures only whether the views actually depict the same
    thing in the same place.
    """
    ref = highpass(plate.astype(np.float32).mean(axis=2))
    if weights is None:
        weights = [1.0] * len(views)
    total, n = 0.0, 0.0
    for v, m, wt in zip(views, masks, weights):
        if m.sum() < 64:
            continue
        vh = highpass(v.astype(np.float32).mean(axis=2))
        a = ref[m]
        b = vh[m]
        a = a - a.mean()
        b = b - b.mean()
        # A featureless patch correlates with anything; it is not evidence.
        if a.std() < 4.0 or b.std() < 4.0:
            continue
        denom = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
        if denom < 1e-6:
            continue
        # Weight the fit exactly as the composite is weighted. Leaving the
        # objective unweighted while weighting the median let re-admitted
        # oblique views drag the plane around with full authority, which made
        # the fits noticeably worse than when those views were simply rejected.
        total += wt * float((a * b).sum()) / denom
        n += wt
    if n < 1.5:
        return -1.0
    return total / n


def sweep(f: dict, sample_fn, axis: str = "offset", n: int = 11):
    """Objective value along one axis, for diagnosing whether the fit has any
    signal to work with at all.

    Worth running before trusting any result from `fit`. On Freiestrasse the
    offset sweep came back flat — 0.271 to 0.300 across the full ±2.5 m range,
    with no peak — which is why the search pinned at its bounds on every façade.
    It was not choosing badly; there was nothing to choose.
    """
    from clean_plate import median_composite
    rows = []
    for value in np.linspace(-OFFSET_RANGE, OFFSET_RANGE, n):
        args = {"offset": 0.0, "yaw": 0.0, "base_dy": 0.0}
        args[axis if axis != "yaw" else "yaw"] = value if axis != "yaw" else math.radians(value)
        Px, Py, Pz = plane_grid(f, args["offset"], args["yaw"], args["base_dy"],
                                FIT_PPM, FIT_HEIGHT)
        views, masks, weights = sample_fn(Px, Py, Pz)
        if len(views) < 3:
            rows.append((value, None, len(views)))
            continue
        plate = median_composite(views, masks, weights)
        rows.append((value, agreement(views, masks, plate, weights), len(views)))
    return rows


def fit(f: dict, sample_fn, verbose: bool = False):
    """Coarse-to-fine search over (offset, yaw, base_dy).

    `sample_fn(Px, Py, Pz) -> (views, masks, weights)` does the projection; keeping it a
    callback means this module knows nothing about camera models.
    """
    best = (None, -1.0)

    def evaluate(offset, yaw, dy):
        Px, Py, Pz = plane_grid(f, offset, yaw, dy, FIT_PPM, FIT_HEIGHT)
        views, masks, weights = sample_fn(Px, Py, Pz)
        if len(views) < 3:
            return None, -1.0
        from clean_plate import median_composite
        plate = median_composite(views, masks, weights)
        coverage = float(np.mean([m.mean() for m in masks]))
        score = agreement(views, masks, plate, weights)
        if score < 0:
            return None, -1.0
        # Coverage matters as a tiebreak only: a plane that sees more of the
        # wall is preferable, but it must not outweigh actually agreeing.
        return plate, score * min(1.0, coverage / 0.7)

    # Coarse pass over offset and yaw, which dominate; then refine with the
    # vertical term, which is a smaller correction.
    for offset in np.linspace(-OFFSET_RANGE, OFFSET_RANGE, 13):
        for yaw in np.linspace(-YAW_RANGE, YAW_RANGE, 7):
            _, score = evaluate(offset, yaw, 0.0)
            if score > best[1]:
                best = ((offset, yaw, 0.0), score)

    if best[0] is None:
        return None, -1.0

    o0, y0, _ = best[0]
    step_o = 2 * OFFSET_RANGE / 12
    step_y = 2 * YAW_RANGE / 6
    for offset in np.clip(np.linspace(o0 - step_o, o0 + step_o, 5),
                          -OFFSET_RANGE, OFFSET_RANGE):
        for yaw in np.clip(np.linspace(y0 - step_y, y0 + step_y, 5),
                           -YAW_RANGE, YAW_RANGE):
            for dy in np.linspace(-1.5, 1.5, 7):
                _, score = evaluate(offset, yaw, dy)
                if score > best[1]:
                    best = ((offset, yaw, dy), score)

    if verbose:
        o, y, d = best[0]
        print(f"      plane fit: offset {o:+.2f} m, yaw {math.degrees(y):+.1f}°, "
              f"base {d:+.2f} m, sharpness {best[1]:.2f}")
    return best
