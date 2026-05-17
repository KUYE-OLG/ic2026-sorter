"""Geometry-based shape classifier — fallback for unknown shapes.

When YOLO can't confidently classify a shape (confidence < 0.85, or
shape not in training set), extract geometric features via OpenCV and
match against a template library of 10+ shapes.

This is the code implementation of the "未知形状兜底" strategy from T5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GeometryFeatures:
    vertices: int           # approxPolyDP vertex count (circle → 0 in practice)
    form_factor: float      # perimeter² / (4π × area), ~1.0 for circle
    convex: bool            # isContourConvex
    aspect_ratio: float     # bbox w/h
    solidity: float         # area / convexHull area
    hu_moments: tuple[float, ...]  # Hu moments (7 values, log-transformed)


# Template library: geometric signature for each supported shape
# Scores: vertices=3pts, form_factor=2pts, convex=1pt, aspect_ratio=1pt
SHAPE_TEMPLATES: dict[str, dict[str, Any]] = {
    "cube":         {"vertices": 4, "ff_range": (1.20, 1.35), "convex": True,  "ar_range": (0.85, 1.15)},
    "cuboid":       {"vertices": 4, "ff_range": (1.25, 2.00), "convex": True,  "ar_range": (1.30, 3.00)},
    "cylinder":     {"vertices": 0, "ff_range": (0.90, 1.10), "convex": True,  "ar_range": (0.85, 1.15)},
    "sphere":       {"vertices": 0, "ff_range": (0.90, 1.10), "convex": True,  "ar_range": (0.90, 1.10)},
    "tri_prism":    {"vertices": 3, "ff_range": (1.20, 1.50), "convex": True,  "ar_range": (0.85, 1.15)},
    "penta_prism":  {"vertices": 5, "ff_range": (1.10, 1.40), "convex": True,  "ar_range": (0.85, 1.15)},
    "hexa_prism":   {"vertices": 6, "ff_range": (1.10, 1.40), "convex": True,  "ar_range": (0.85, 1.15)},
    "star":         {"vertices": 10,"ff_range": (1.80, 3.00), "convex": False, "ar_range": (0.85, 1.15)},
    "thin_plate":   {"vertices": 4, "ff_range": (2.50, 8.00), "convex": True,  "ar_range": (2.00, 8.00)},
    "arrow":        {"vertices": 7, "ff_range": (1.80, 3.50), "convex": False, "ar_range": (1.50, 3.00)},
}


def extract_geometry(contour: Any) -> GeometryFeatures:
    """Extract geometric features from an OpenCV contour.

    Args:
        contour: OpenCV contour (numpy array of points).

    Returns:
        GeometryFeatures with vertices, form_factor, convex, aspect_ratio, solidity, hu.
    """
    import cv2
    import numpy as np

    # Vertices via polygon approximation
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    vertices = len(approx)

    # Form factor: perimeter² / (4π × area)
    area = cv2.contourArea(contour)
    form_factor = (peri * peri) / (4 * math.pi * area) if area > 0 else 999

    # Convexity
    convex = cv2.isContourConvex(contour)

    # Aspect ratio from minAreaRect
    rect = cv2.minAreaRect(contour)
    w, h = rect[1]
    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 999

    # Solidity: area / convexHull area
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    # Hu moments (log-transformed for scale invariance)
    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments)
    hu_log: list[float] = []
    for i in range(7):
        val = hu[i][0]
        hu_log.append(-math.log10(abs(val)) if abs(val) > 1e-10 else 0)

    return GeometryFeatures(
        vertices=vertices,
        form_factor=round(form_factor, 3),
        convex=convex,
        aspect_ratio=round(aspect_ratio, 3),
        solidity=round(solidity, 3),
        hu_moments=tuple(round(h, 3) for h in hu_log),
    )


def match_by_geometry(features: GeometryFeatures) -> tuple[str, float]:
    """Match geometry features to closest shape template.

    Returns (shape_name, confidence_score).
    Score range: 0-7 (3 vertices + 2 form_factor + 1 convex + 1 aspect).

    Args:
        features: Extracted geometric features.

    Returns:
        (shape_name, normalized_confidence) or ("unknown", 0.0).
    """
    best_match = "unknown"
    best_score = 0.0
    max_possible = 7.0

    for name, tpl in SHAPE_TEMPLATES.items():
        score = 0

        # Vertices match (3 pts)
        if features.vertices == tpl["vertices"]:
            score += 3

        # Form factor in range (2 pts)
        lo, hi = tpl["ff_range"]
        if lo <= features.form_factor <= hi:
            score += 2

        # Convexity match (1 pt)
        if features.convex == tpl["convex"]:
            score += 1

        # Aspect ratio in range (1 pt)
        ar_lo, ar_hi = tpl.get("ar_range", (0.5, 2.0))
        if ar_lo <= features.aspect_ratio <= ar_hi:
            score += 1

        if score > best_score:
            best_score = score
            best_match = name

    return best_match, round(best_score / max_possible, 3)
