"""Local polygon-IoU diagnostics and ICDAR 2015 Task 4.1 result packaging."""
import re
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from .geometry import clockwise_quad, polygon


def checked_predictions(records, predictions):
    expected = {record.image_id for record in records}
    missing = expected - set(predictions)
    extra = set(predictions) - expected
    if missing or extra:
        raise ValueError(f"Prediction IDs differ from manifest. "
                         f"Missing ({len(missing)}): {sorted(missing)[:5]}. "
                         f"Extra ({len(extra)}): {sorted(extra)[:5]}.")


def evaluate(records, predictions, iou_threshold, ignore_overlap):
    checked_predictions(records, predictions)
    true_positive = num_truth = num_predictions = ignored_predictions = 0
    for record in records:
        if record.instances is None:
            raise ValueError(f"Ground truth is absent for {record.image_id}.")
        truth = [polygon(points) for points, ignored in record.instances if not ignored]
        ignored_truth = [polygon(points) for points, ignored in record.instances if ignored]
        num_truth += len(truth)
        matched = set()
        detections = sorted(predictions[record.image_id], key=lambda item: item["score"], reverse=True)
        for item in detections:
            detected = polygon(item["points"])
            if any(detected.intersection(shape).area / detected.area > ignore_overlap
                   for shape in ignored_truth):
                ignored_predictions += 1
                continue
            num_predictions += 1
            best_index = None
            best_iou = -1.0
            for index, target in enumerate(truth):
                if index in matched:
                    continue
                intersection = detected.intersection(target).area
                iou = intersection / (detected.area + target.area - intersection)
                if iou >= iou_threshold and iou > best_iou:
                    best_index, best_iou = index, iou
            if best_index is not None:
                matched.add(best_index)
                true_positive += 1
    precision = true_positive / num_predictions if num_predictions else 0.0
    recall = true_positive / num_truth if num_truth else 0.0
    f_measure = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"protocol": "local_polygon_iou_score_ordered_greedy_v1",
            "iou_threshold": iou_threshold, "ignore_overlap": ignore_overlap,
            "images": len(records), "tp": true_positive,
            "fp": num_predictions - true_positive, "fn": num_truth - true_positive,
            "ignored_detections": ignored_predictions,
            "precision": precision, "recall": recall, "f_measure": f_measure}


def export_icdar2015(path, records, predictions):
    checked_predictions(records, predictions)
    entries = {}
    numeric_ids = set()
    for record in records:
        image_id = record.image_id
        match = re.fullmatch(r"img_(\d+)", image_id)
        if not match or int(match.group(1)) in numeric_ids:
            raise ValueError("ICDAR 2015 export requires unique img_<number> IDs.")
        numeric_ids.add(int(match.group(1)))
        lines = []
        for detection in predictions[image_id]:
            points = np.rint(np.asarray(detection["points"])).astype(np.int64)
            if points.shape != (4, 2):
                raise ValueError("ICDAR 2015 export requires quad predictions.")
            try:
                polygon(points)
            except ValueError:
                # Quantization at image edges can collapse otherwise valid quads.
                continue
            lines.append(",".join(map(str, clockwise_quad(points).ravel())))
        entries[f"res_{image_id}.txt"] = ("\r\n".join(lines) + "\r\n") if lines else ""
    with ZipFile(path, "x", ZIP_DEFLATED) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
