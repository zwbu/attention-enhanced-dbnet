"""Run training, inference and evaluation via python -m attndb."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import random

import cv2
import numpy as np
import shapely
import torch
from torch.utils.data import DataLoader

from .data import TextDataset, convert_icdar, image_tensor, read_image, read_manifest
from .evaluation import evaluate, export_icdar2015
from .geometry import decode
from .losses import db_loss
from .model import AttentionDBNet, VARIANTS


IMPLEMENTATION = "jei-263038-reference-v0.1"


def read_json(path):
    return Path(path).read_text(encoding="utf-8-sig")


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def file_hash(path):
    with Path(path).open("rb") as stream:
        digest = sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def choose_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this PyTorch environment.")
    return torch.device(name)


def train(args):
    config = json.loads(read_json(args.config))
    options = config["training"]
    epochs = args.epochs if args.epochs is not None else options["epochs"]
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("Training requires a positive epoch count.")
    options["epochs"] = epochs
    batch_size = options["batch_size"]
    workers = options["workers"]
    size = tuple(config["image_size"])
    if batch_size < 1 or workers < 0 or min(size) < 64 or any(side % 32 for side in size):
        raise ValueError("Require positive batch size, nonnegative workers and image dimensions "
                         "that are multiples of 32 and at least 64.")
    if len(config["model"]["pyramid_channels"]) != 4:
        raise ValueError("The reference model requires four pyramid levels.")
    device = choose_device(args.device)
    seed = args.seed if args.seed is not None else options["seed"]
    options["seed"] = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    records = read_manifest(args.manifest)
    dataset = TextDataset(records, size, config["targets"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=workers, pin_memory=device.type == "cuda",
                        generator=torch.Generator().manual_seed(seed))
    model = AttentionDBNet(args.variant, **config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options["lr"],
                                  weight_decay=options["weight_decay"])
    run = Path(args.out)
    run.mkdir(parents=True, exist_ok=False)
    provenance = {"implementation": IMPLEMENTATION, "variant": args.variant,
                  "train_manifest_sha256": file_hash(args.manifest), "images": len(records),
                  "device": str(device), "python": platform.python_version(),
                  "pytorch": str(torch.__version__), "cuda": torch.version.cuda,
                  "platform": platform.platform(), "numpy": np.__version__,
                  "opencv": cv2.__version__, "shapely": shapely.__version__}
    if device.type == "cuda":
        provenance["gpu"] = torch.cuda.get_device_name(device)
    write_json(run / "run.json", {**provenance, "config": config})
    model.train()
    with (run / "loss.jsonl").open("x", encoding="utf-8") as log:
        for epoch in range(1, epochs + 1):
            total = 0.0
            for image, targets in loader:
                image = image.to(device, non_blocking=True)
                targets = {key: value.to(device, non_blocking=True) for key, value in targets.items()}
                optimizer.zero_grad(set_to_none=True)
                loss = db_loss(model(image), targets, **config["loss"])
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite training loss at epoch {epoch}.")
                loss.backward()
                optimizer.step()
                total += loss.item() * len(image)
            metrics = {"epoch": epoch, "train_loss": total / len(dataset)}
            print(json.dumps(metrics), flush=True)
            log.write(json.dumps(metrics) + "\n")
            log.flush()

    checkpoint = run / "last.pt"
    temporary = run / "last.pt.tmp"
    torch.save({"implementation": IMPLEMENTATION, "config": config, "variant": args.variant,
                "epoch": epochs, "model": model.state_dict()}, temporary)
    temporary.replace(checkpoint)
    write_json(run / "checkpoint.json", {"file": checkpoint.name, "sha256": file_hash(checkpoint)})


def predict(args):
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    if checkpoint.get("implementation") != IMPLEMENTATION:
        raise ValueError("Expected a checkpoint produced by this reference package.")
    config = checkpoint["config"]
    model = AttentionDBNet(checkpoint["variant"], **config["model"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    size = tuple(config["image_size"])
    settings = dict(config["postprocess"])
    map_source = settings.pop("map_source")
    records = read_manifest(args.manifest)
    overlay = Path(args.overlays) if args.overlays else None
    if overlay is not None:
        overlay.mkdir(parents=True, exist_ok=False)
    detections = {}
    with torch.inference_mode():
        for record in records:
            image = read_image(record.image)
            output = model(image_tensor(image, size).unsqueeze(0).to(device))
            score_map = output[map_source][0, 0].cpu().numpy()
            probability = output["probability"][0, 0].cpu().numpy()
            found = decode(score_map, probability, image.shape[:2], **settings)
            detections[record.image_id] = found
            if overlay is not None:
                for item in found:
                    vertices = np.rint(item["points"]).astype(np.int32)
                    cv2.polylines(image, [vertices], True, (255, 80, 0), 2, cv2.LINE_AA)
                ok, encoded = cv2.imencode(".png", image)
                if not ok:
                    raise RuntimeError(f"Could not encode overlay for {record.image_id}.")
                encoded.tofile(overlay / f"{record.image_id}.png")
    write_json(args.out, {"implementation": IMPLEMENTATION, "benchmark": config["benchmark"],
                          "variant": checkpoint["variant"], "epoch": checkpoint["epoch"],
                          "checkpoint_sha256": file_hash(args.checkpoint),
                          "inference_manifest_sha256": file_hash(args.manifest),
                          "geometry": settings["geometry"], "detections": detections})


def score(args):
    records = read_manifest(args.manifest)
    result = json.loads(read_json(args.predictions))
    benchmark = result["benchmark"]
    iou = args.iou
    if iou is None:
        iou = 0.6 if benchmark == "ctst1600" else 0.5
    metrics = evaluate(records, result["detections"], iou, args.ignore_overlap)
    metrics.update({"benchmark": benchmark, "variant": result["variant"],
                    "ground_truth_manifest_sha256": file_hash(args.manifest),
                    "predictions_sha256": file_hash(args.predictions),
                    "checkpoint_sha256": result["checkpoint_sha256"]})
    if args.out:
        write_json(args.out, metrics)
    print(json.dumps(metrics, indent=2))


def export(args):
    records = read_manifest(args.manifest)
    result = json.loads(read_json(args.predictions))
    if result["benchmark"] != "icdar2015" or result["geometry"] != "quad":
        raise ValueError("Use ICDAR 2015 quadrilateral predictions.")
    if {record.image_id for record in records} != {f"img_{index}" for index in range(1, 501)}:
        raise ValueError("Expected the 500 ICDAR 2015 test images (img_1 through img_500).")
    if result["inference_manifest_sha256"] != file_hash(args.manifest):
        raise ValueError("Use the original manifest from inference.")
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    export_icdar2015(path, records, result["detections"])


def convert(args):
    rows = convert_icdar(args.benchmark, args.images, args.labels, args.out)
    write_json(args.out, rows)
    print(f"Wrote {len(rows)} image records.")


def unit_interval(value):
    value = float(value)
    if not 0 < value <= 1:
        raise argparse.ArgumentTypeError("Expected a number in (0, 1].")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-icdar", help="Convert native ICDAR annotations.")
    prepare.add_argument("benchmark", choices=("icdar2015", "icdar2013"))
    prepare.add_argument("--images", required=True)
    prepare.add_argument("--labels", help="Omit for image-only inference.")
    prepare.add_argument("--out", required=True)
    prepare.set_defaults(handler=convert)

    training = commands.add_parser("train", help="Train one architecture variant from scratch.")
    training.add_argument("--config", required=True)
    training.add_argument("--manifest", required=True)
    training.add_argument("--variant", choices=VARIANTS, default="full")
    training.add_argument("--epochs", type=int)
    training.add_argument("--seed", type=int)
    training.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    training.add_argument("--out", required=True, help="New output directory.")
    training.set_defaults(handler=train)

    inference = commands.add_parser("predict", help="Run inference from a reference checkpoint.")
    inference.add_argument("--checkpoint", required=True)
    inference.add_argument("--manifest", required=True)
    inference.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    inference.add_argument("--overlays", help="Optional new output directory for visualizations.")
    inference.add_argument("--out", required=True, help="New JSON file.")
    inference.set_defaults(handler=predict)

    evaluation = commands.add_parser("evaluate", help="Run local polygon-IoU diagnostics.")
    evaluation.add_argument("--manifest", required=True)
    evaluation.add_argument("--predictions", required=True)
    evaluation.add_argument("--iou", type=unit_interval)
    evaluation.add_argument("--ignore-overlap", type=unit_interval, default=0.5)
    evaluation.add_argument("--out", help="Optional new JSON file.")
    evaluation.set_defaults(handler=score)

    packing = commands.add_parser("export-icdar2015", help="Package ICDAR 2015 Task 4.1 results.")
    packing.add_argument("--manifest", required=True)
    packing.add_argument("--predictions", required=True)
    packing.add_argument("--out", required=True, help="New ZIP file.")
    packing.set_defaults(handler=export)
    args = parser.parse_args()
    try:
        args.handler(args)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
