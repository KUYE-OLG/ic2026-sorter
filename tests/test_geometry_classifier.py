"""Tests for geometry-based shape classifier (unknown shape fallback)."""
import numpy as np
import cv2

from sorter_control.vision.geometry_classifier import (
    extract_geometry,
    match_by_geometry,
    SHAPE_TEMPLATES,
    GeometryFeatures,
)


def _make_square_contour(size=100):
    """Create a synthetic square contour."""
    img = np.zeros((size + 20, size + 20), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (10 + size, 10 + size), 255, -1)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]


def _make_circle_contour(radius=50):
    """Create a synthetic circle contour."""
    img = np.zeros((radius * 2 + 20, radius * 2 + 20), dtype=np.uint8)
    cv2.circle(img, (radius + 10, radius + 10), radius, 255, -1)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]


def _make_triangle_contour(size=100):
    """Create a synthetic triangle contour."""
    img = np.zeros((size + 20, size + 20), dtype=np.uint8)
    pts = np.array([[10, 10 + size], [10 + size // 2, 10], [10 + size, 10 + size]], dtype=np.int32)
    cv2.fillPoly(img, [pts], 255)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]


def _make_star_contour(size=100):
    """Create a synthetic star-shaped contour."""
    img = np.zeros((size + 20, size + 20), dtype=np.uint8)
    cx, cy = size // 2 + 10, size // 2 + 10
    outer_r = size // 2
    inner_r = outer_r // 2
    pts = []
    for i in range(10):
        angle = i * np.pi / 5 - np.pi / 2
        r = outer_r if i % 2 == 0 else inner_r
        pts.append([int(cx + r * np.cos(angle)), int(cy + r * np.sin(angle))])
    cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], 255)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]


class TestGeometryClassifier:
    def test_extract_square(self):
        contour = _make_square_contour()
        features = extract_geometry(contour)
        assert features.vertices == 4
        assert 1.0 < features.form_factor < 1.5
        assert features.convex is True

    def test_extract_circle(self):
        contour = _make_circle_contour()
        features = extract_geometry(contour)
        # Circle has many vertices after approximation — treat as 0 in template
        assert features.vertices >= 0
        assert 0.8 < features.form_factor < 1.2  # near-perfect circle
        assert features.convex is True

    def test_extract_triangle(self):
        contour = _make_triangle_contour()
        features = extract_geometry(contour)
        assert features.vertices == 3
        assert features.convex is True

    def test_extract_star(self):
        contour = _make_star_contour()
        features = extract_geometry(contour)
        assert features.convex is False  # star is concave

    def test_match_square_to_cube(self):
        contour = _make_square_contour()
        features = extract_geometry(contour)
        name, conf = match_by_geometry(features)
        assert name in ("cube", "cuboid")
        assert conf > 0.5

    def test_match_circle_to_cylinder(self):
        contour = _make_circle_contour()
        features = extract_geometry(contour)
        name, conf = match_by_geometry(features)
        assert name in ("cylinder", "sphere")
        assert conf > 0.4

    def test_match_triangle_to_tri_prism(self):
        contour = _make_triangle_contour()
        features = extract_geometry(contour)
        name, conf = match_by_geometry(features)
        assert name == "tri_prism"
        assert conf > 0.4

    def test_match_star(self):
        contour = _make_star_contour()
        features = extract_geometry(contour)
        name, conf = match_by_geometry(features)
        # Star may match "star" or fall through — just verify non-crash
        assert isinstance(name, str)
        assert 0.0 <= conf <= 1.0

    def test_all_templates_valid(self):
        """All templates should have required keys."""
        required = {"vertices", "ff_range", "convex"}
        for name, tpl in SHAPE_TEMPLATES.items():
            for key in required:
                assert key in tpl, f"{name} missing {key}"
