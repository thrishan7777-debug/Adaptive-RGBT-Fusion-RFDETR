
import argparse
import json
from pathlib import Path

import torch


def validate_coco_dataset(dataset_dir):

    dataset_dir = Path(dataset_dir)

    annotation_file = (
        dataset_dir / "_annotations.coco.json"
    )

    if not annotation_file.exists():
        raise FileNotFoundError(annotation_file)

    with open(annotation_file, "r", encoding="utf-8") as file:
        coco = json.load(file)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    if not images:
        raise ValueError("COCO dataset has no images.")

    if not categories:
        raise ValueError("COCO dataset has no categories.")

    missing_images = []

    for image in images:

        image_path = dataset_dir / image["file_name"]

        if not image_path.exists():
            missing_images.append(str(image_path))

    if missing_images:
        raise FileNotFoundError(
            "Missing fused images:\n"
            + "\n".join(missing_images[:10])
        )

    image_ids = {
        image["id"] for image in images
    }

    for annotation in annotations:

        if annotation["image_id"] not in image_ids:
            raise ValueError(
                "Annotation references an unknown image ID."
            )

        bbox = annotation["bbox"]

        if len(bbox) != 4:
            raise ValueError("Invalid COCO bounding box.")

        x, y, width, height = bbox

        if width <= 0 or height <= 0:
            raise ValueError("Invalid bounding box dimensions.")

    print("COCO validation successful.")
    print("Images:", len(images))
    print("Annotations:", len(annotations))
    print("Categories:", len(categories))

    return coco


def train(args):

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available() and args.device == "cuda":
        raise RuntimeError("CUDA requested but unavailable.")

    dataset_dir = Path(args.dataset)

    validate_coco_dataset(dataset_dir)

    try:
        from rfdetr import RFDETRLarge
    except ImportError as error:
        raise ImportError(
            "RF-DETR is not installed. Install a compatible "
            "RF-DETR version in your environment."
        ) from error

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = RFDETRLarge()

    print("\nRF-DETR-L configuration")
    print("Resolution:", args.resolution)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    print("Gradient accumulation:", args.grad_accum_steps)
    print("EMA:", args.use_ema)

    print("\nStarting detector training...")

    model.train(
        dataset_dir=str(dataset_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        resolution=args.resolution,
        output_dir=str(output_dir),
        device=args.device,
        use_ema=args.use_ema
    )

    print("\nRF-DETR-L training finished.")
    print("Output directory:", output_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)

    parser.add_argument(
        "--epochs",
        type=int,
        default=30
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=4
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=1024
    )

    parser.add_argument(
        "--device",
        default="cuda"
    )

    parser.add_argument(
        "--use-ema",
        action="store_true"
    )

    parser.add_argument(
        "--output",
        default="outputs/rfdetr"
    )

    args = parser.parse_args()

    train(args)
