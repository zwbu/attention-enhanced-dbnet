"""Read explicit data splits and convert supported benchmark annotation formats."""
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import polygon, training_maps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class ImageRecord:
    image_id: str
    image: Path
    instances: list | None


def read_manifest(path):
    path = Path(path).resolve()
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: expected a nonempty list of image records.")
    seen = set()
    records = []
    for row in rows:
        image_id = row["id"]
        if not isinstance(image_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", image_id):
            raise ValueError(f"{path}: invalid image ID {image_id!r}.")
        if image_id in seen:
            raise ValueError(f"{path}: duplicate image ID {image_id!r}.")
        seen.add(image_id)
        image = (path.parent / row["image"]).resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        annotations = None
        if "instances" in row:
            if not isinstance(row["instances"], list):
                raise ValueError(f"{image_id}: instances must be a list.")
            annotations = []
            for item in row["instances"]:
                ignored = item.get("ignore", False)
                if not isinstance(ignored, bool):
                    raise ValueError(f"{image_id}: ignore must be true or false.")
                try:
                    points = np.asarray(polygon(item["points"]).exterior.coords[:-1],
                                        dtype=np.float64)
                except ValueError as error:
                    raise ValueError(f"{image_id}: {error}") from error
                annotations.append((points, ignored))
        records.append(ImageRecord(image_id, image, annotations))
    return records


IC13_LINE = re.compile(r"^(-?\d+)\s*[,\s]\s*(-?\d+)\s*[,\s]\s*(-?\d+)"
                       r"\s*[,\s]\s*(-?\d+)(?:[\s,]+(.*))?$")


def read_icdar_label(path, benchmark):
    entries = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            if benchmark == "icdar2015":
                fields = next(csv.reader([line], skipinitialspace=True))
                if len(fields) < 9:
                    raise ValueError("Expected eight coordinates and a transcription.")
                points = np.array([float(field) for field in fields[:8]]).reshape(4, 2)
                text = ",".join(fields[8:]).strip().strip('"')
            else:
                match = IC13_LINE.fullmatch(line.strip())
                if match is None:
                    raise ValueError("Expected an axis-aligned box and optional transcription.")
                left, top, right, bottom = map(int, match.groups()[:4])
                if right <= left or bottom <= top:
                    raise ValueError("Invalid axis-aligned box.")
                points = [[left, top], [right, top], [right, bottom], [left, bottom]]
                text = (match.group(5) or "").strip().strip('"')
            polygon(points)
        except ValueError as error:
            raise ValueError(f"{path}:{number}: {error}") from error
        entries.append({"points": np.asarray(points).tolist(), "ignore": text == "###"})
    return entries


def convert_icdar(benchmark, images, labels, output):
    images = Path(images).resolve()
    output = Path(output).resolve()
    rows = []
    for image in sorted(images.iterdir()):
        if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        # Use an absolute path only when Windows paths are on different drives.
        image_path = image.as_posix()
        if image.anchor == output.anchor:
            image_path = Path(os.path.relpath(image, output.parent)).as_posix()
        row = {"id": image.stem, "image": image_path}
        if labels is not None:
            label_path = Path(labels) / f"gt_{image.stem}.txt"
            row["instances"] = read_icdar_label(label_path, benchmark)
        rows.append(row)
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Image directory is empty or contains duplicate image stems.")
    return rows


def read_image(path):
    flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if bgr is None:
        raise ValueError(f"Could not decode image: {path}")
    return bgr


def image_tensor(bgr, size):
    height, width = size
    rgb = cv2.cvtColor(cv2.resize(bgr, (width, height)), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().div_(127.5).sub_(1)


class TextDataset(Dataset):
    def __init__(self, records, size, target_settings):
        if any(record.instances is None for record in records):
            raise ValueError("Training requires instances for every image. Use [] for negatives.")
        self.records = records
        self.size = tuple(size)
        self.target_settings = target_settings

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        bgr = read_image(record.image)
        height, width = bgr.shape[:2]
        new_height, new_width = self.size
        scale = np.array([new_width / width, new_height / height])
        scaled = [(points * scale, ignored) for points, ignored in record.instances]
        targets = training_maps(scaled, self.size, **self.target_settings)
        return image_tensor(bgr, self.size), targets
