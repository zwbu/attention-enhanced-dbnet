"""Synthetic regression checks. These fixtures never represent paper results."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

import cv2
import numpy as np
import torch

from attndb.data import ImageRecord, read_icdar_label
from attndb.evaluation import evaluate
from attndb.geometry import decode, training_maps
from attndb.model import AttentionDBNet, VARIANTS


ROOT = Path(__file__).resolve().parents[1]
RECT = [[9, 10], [37, 10], [37, 30], [9, 30]]
IGNORE = [[44, 41], [58, 41], [58, 57], [44, 57]]


def small_config():
    config = json.loads((ROOT / "configs/icdar2015.json").read_text())
    config["image_size"] = [64, 64]
    config["model"].update(stem_channels=8, pyramid_channels=[8, 12, 16, 20], neck_channels=8)
    config["training"]["batch_size"] = 2
    return config


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_variants_and_gradients(self):
        image = torch.randn(2, 3, 65, 97)
        for variant, (scaf, feature) in VARIANTS.items():
            with self.subTest(variant=variant):
                model = AttentionDBNet(variant, **small_config()["model"])
                output = model(image)
                self.assertEqual(tuple(output["binary"].shape), (2, 1, 65, 97))
                self.assertEqual(not isinstance(model.scaf, torch.nn.Identity), scaf)
                self.assertEqual(len(model.fusion), 3 if feature else 0)
                for name in ("probability", "threshold", "binary"):
                    self.assertTrue(torch.all((output[name] >= 0) & (output[name] <= 1)))
                output["binary"].mean().backward()
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_polygon_targets_and_decoder(self):
        targets = training_maps([(RECT, False), (IGNORE, True)], (64, 64),
                                **small_config()["targets"])
        self.assertEqual(targets["shrink"][20, 20], 1)
        self.assertEqual(targets["shrink"][0, 0], 0)
        self.assertEqual(targets["valid"][50, 50], 0)
        self.assertEqual(targets["border_mask"][50, 50], 0)
        self.assertGreater(targets["border"][10, 20], 0.6)
        binary = targets["shrink"]
        for geometry in ("quad", "polygon"):
            settings = dict(small_config()["postprocess"])
            settings.pop("map_source")
            settings["geometry"] = geometry
            detections = decode(binary, binary, (128, 96), **settings)
            self.assertEqual(len(detections), 1)
            if geometry == "quad":
                self.assertEqual(len(detections[0]["points"]), 4)
            self.assertGreater(np.mean(detections[0]["points"], axis=0)[0], 25)

    def test_evaluation_ignores_and_deduplicates(self):
        records = [ImageRecord("img_1", Path("unused"), [(RECT, False), (IGNORE, True)])]
        detections = {"img_1": [{"points": points, "score": score}
                                for points, score in [(RECT, 0.9), (RECT, 0.8), (IGNORE, 0.7)]]}
        metrics = evaluate(records, detections, 0.6, 0.5)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 1, 0))
        self.assertEqual(metrics["ignored_detections"], 1)
        self.assertAlmostEqual(metrics["f_measure"], 2 / 3)
        with self.assertRaisesRegex(ValueError, "Missing"):
            evaluate(records, {}, 0.6, 0.5)

    def test_native_annotation_readers(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "gt_img_1.txt"
            path.write_text("9 10 37 30 \"EXIT 4\"\n44,41,58,57,\"###\"\n", encoding="utf-8")
            labels = read_icdar_label(path, "icdar2013")
            self.assertEqual(labels[0]["points"], RECT)
            self.assertTrue(labels[1]["ignore"])
            path.write_text("9,10,37,10,37,30,9,30,\"EXIT, 4\"\n", encoding="utf-8-sig")
            labels = read_icdar_label(path, "icdar2015")
            self.assertEqual(labels[0]["points"], RECT)
            self.assertFalse(labels[0]["ignore"])

    def test_cli_cpu_train_predict_evaluate_export(self):
        def cli(*args):
            return subprocess.run([sys.executable, "-m", "attndb", *map(str, args)],
                                  cwd=ROOT, check=True, text=True, capture_output=True)

        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            images = path / "images"
            labels = path / "labels"
            images.mkdir()
            labels.mkdir()
            bgr = np.zeros((64, 64, 3), np.uint8)
            cv2.rectangle(bgr, (9, 10), (37, 30), (255, 255, 255), -1)
            for i in (1, 2):
                self.assertTrue(cv2.imwrite(str(images / f"img_{i}.png"), bgr))
                label = "9,10,37,10,37,30,9,30,EXIT\n" if i == 1 else ""
                (labels / f"gt_img_{i}.txt").write_text(label)
            manifest = path / "set.json"
            config = path / "config.json"
            config.write_text(json.dumps(small_config()))
            cli("prepare-icdar", "icdar2015", "--images", images, "--labels", labels, "--out", manifest)
            run = path / "run"
            cli("train", "--config", config, "--manifest", manifest, "--variant", "full",
                "--epochs", "1", "--device", "cpu", "--out", run)
            self.assertTrue((run / "last.pt").is_file())
            self.assertTrue(np.isfinite(json.loads((run / "loss.jsonl").read_text())["train_loss"]))
            predictions = path / "predictions.json"
            cli("predict", "--checkpoint", run / "last.pt", "--manifest", manifest,
                "--device", "cpu", "--out", predictions, "--overlays", path / "overlays")
            self.assertEqual(len(list((path / "overlays").glob("*.png"))), 2)
            metrics = json.loads(cli("evaluate", "--manifest", manifest, "--predictions", predictions).stdout)
            self.assertEqual(metrics["images"], 2)
            self.assertEqual(metrics["variant"], "full")
            # Format-only exercise of the 500-ID exporter, with no benchmark data.
            from hashlib import sha256
            manifest.write_text(json.dumps([{"id": f"img_{i}", "image": "images/img_1.png"}
                                            for i in range(1, 501)]))
            result = json.loads(predictions.read_text())
            result["inference_manifest_sha256"] = sha256(manifest.read_bytes()).hexdigest()
            result["detections"] = {f"img_{i}": [] for i in range(1, 501)}
            result["detections"]["img_1"] = [{"points": RECT, "score": 1.0}]
            predictions.write_text(json.dumps(result))
            archive = path / "rrc.zip"
            cli("export-icdar2015", "--manifest", manifest, "--predictions", predictions, "--out", archive)
            with ZipFile(archive) as zipped:
                self.assertEqual(len(zipped.namelist()), 500)
                self.assertEqual(zipped.read("res_img_1.txt"), b"9,10,37,10,37,30,9,30\r\n")
                self.assertEqual(zipped.read("res_img_500.txt"), b"")


if __name__ == "__main__":
    unittest.main()
