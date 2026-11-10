"""Polygon validation, DB training maps and contour postprocessing."""
import cv2
import numpy as np
from shapely.geometry import Polygon


def polygon(points):
    coords = np.asarray(points, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[0] < 3 or coords.shape[1] != 2:
        raise ValueError("A polygon needs at least three [x, y] vertices.")
    if not np.isfinite(coords).all():
        raise ValueError("A polygon contains a non-finite coordinate.")
    shape = Polygon(coords)
    if not shape.is_valid or shape.area <= 0:
        raise ValueError("A polygon is self-intersecting or degenerate.")
    return shape


def polygon_parts(shape):
    if shape.is_empty:
        return []
    if shape.geom_type == "Polygon":
        return [shape]
    return [part for part in getattr(shape, "geoms", []) if part.geom_type == "Polygon"]


def paint(canvas, shape, value, origin=(0, 0)):
    for part in polygon_parts(shape):
        exterior = np.rint(np.asarray(part.exterior.coords) - origin).astype(np.int32)
        cv2.fillPoly(canvas, [exterior], value)


def border_target(shape, expanded, distance, border, border_mask):
    height, width = border.shape
    x_min, y_min, x_max, y_max = expanded.bounds
    left = max(0, int(np.floor(x_min)))
    top = max(0, int(np.floor(y_min)))
    right = min(width, int(np.ceil(x_max)) + 1)
    bottom = min(height, int(np.ceil(y_max)) + 1)
    if left >= right or top >= bottom:
        return
    region = np.zeros((bottom - top, right - left), np.uint8)
    paint(region, expanded, 1, (left, top))
    ys, xs = np.nonzero(region)
    if not len(xs):
        return
    x = xs + left
    y = ys + top
    px = x.astype(np.float64)
    py = y.astype(np.float64)
    squared_distance = np.full(len(x), np.inf)
    vertices = np.asarray(shape.exterior.coords)
    for (ax, ay), (bx, by) in zip(vertices[:-1], vertices[1:]):
        dx, dy = bx - ax, by - ay
        length_squared = dx * dx + dy * dy
        if length_squared == 0:
            continue
        projection = np.clip(((px - ax) * dx + (py - ay) * dy) / length_squared, 0, 1)
        squared_distance = np.minimum(squared_distance, (px - ax - projection * dx) ** 2
                                      + (py - ay - projection * dy) ** 2)
    values = np.clip(1 - np.sqrt(squared_distance) / distance, 0, 1)
    border[y, x] = np.maximum(border[y, x], values)
    border_mask[y, x] = 1


def training_maps(instances, size, shrink_ratio, min_text_size, threshold_min, threshold_max):
    height, width = size
    shrink = np.zeros(size, np.float32)
    valid = np.ones(size, np.float32)
    border = np.zeros(size, np.float32)
    border_mask = np.zeros(size, np.float32)
    frame = Polygon([(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)])

    for points, ignored in instances:
        for shape in polygon_parts(polygon(points).intersection(frame)):
            x_min, y_min, x_max, y_max = shape.bounds
            if ignored or min(x_max - x_min, y_max - y_min) < min_text_size:
                paint(valid, shape, 0)
                continue
            distance = shape.area * (1 - shrink_ratio ** 2) / shape.length
            inner = shape.buffer(-distance)
            if inner.is_empty:
                paint(valid, shape, 0)
                continue
            paint(shrink, inner, 1)
            border_target(shape, shape.buffer(distance), distance, border, border_mask)

    return {"shrink": shrink, "valid": valid,
            "border": threshold_min + border * (threshold_max - threshold_min),
            "border_mask": border_mask * valid}


def clockwise_quad(points):
    points = np.asarray(points)
    start = min(range(len(points)), key=lambda index: (points[index].sum(), points[index, 0]))
    points = np.roll(points, -start, axis=0)
    x, y = points.T
    if (x * np.roll(y, -1) - y * np.roll(x, -1)).sum() < 0:
        points = points[[0, 3, 2, 1]]
    return points


def decode(score_map, probability, original_size, pixel_threshold, score_threshold,
           unclip_ratio, min_side, max_candidates, geometry):
    height, width = score_map.shape
    original_height, original_width = original_size
    contours, _ = cv2.findContours(np.uint8(score_map >= pixel_threshold),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_candidates]
    detections = []
    for contour in contours:
        if len(contour) < 3 or min(cv2.minAreaRect(contour)[1]) < min_side:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [contour - np.array([x, y], np.int32)], 1)
        score = float(probability[y:y+h, x:x+w][mask != 0].mean())
        if score < score_threshold:
            continue
        points = cv2.approxPolyDP(contour, 0.005 * cv2.arcLength(contour, True), True).reshape(-1, 2)
        try:
            shape = polygon(points)
        except ValueError:
            continue
        expanded = shape.buffer(unclip_ratio * shape.area / shape.length)
        if expanded.geom_type != "Polygon" or expanded.is_empty:
            continue
        coords = np.asarray(expanded.exterior.coords[:-1], dtype=np.float32)
        if geometry == "quad":
            coords = cv2.boxPoints(cv2.minAreaRect(coords))
        coords[:, 0] = np.clip(coords[:, 0] * original_width / width, 0, original_width - 1)
        coords[:, 1] = np.clip(coords[:, 1] * original_height / height, 0, original_height - 1)
        if geometry == "quad":
            coords = clockwise_quad(coords)
        try:
            polygon(coords)
        except ValueError:
            continue
        detections.append({"points": coords.tolist(), "score": score})
    return sorted(detections, key=lambda detection: detection["score"], reverse=True)
