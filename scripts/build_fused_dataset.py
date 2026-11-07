
import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms.functional as TF

from src.fusion.fusion_model import AdaptiveRGBThermalFusion


def load_rgb(path, device):
    image = Image.open(path).convert("RGB")
    return TF.to_tensor(image).unsqueeze(0).to(device)


def load_thermal(path, device):
    image = Image.open(path).convert("L")
    return TF.to_tensor(image).unsqueeze(0).to(device)


def main(args):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    model = AdaptiveRGBThermalFusion().to(device)

    state = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state)
    model.eval()

    source_json = Path(args.coco_json)

    with open(source_json, "r", encoding="utf-8") as file:
        coco = json.load(file)

    source_images = {
        image["id"]: image
        for image in coco["images"]
    }

    output_dir = Path(args.output)
    image_dir = output_dir / "images"

    image_dir.mkdir(parents=True, exist_ok=True)

    new_images = []
    image_mapping = {}

    with open(args.pairs_csv, "r", encoding="utf-8") as file:

        reader = csv.DictReader(file)

        required = {"image_id", "rgb_path", "thermal_path"}

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                "CSV must contain image_id, rgb_path, thermal_path"
            )

        rows = list(reader)

    if not rows:
        raise ValueError("Pair manifest is empty.")

    seen_ids = set()

    with torch.inference_mode():

        for index, row in enumerate(rows, start=1):

            image_id = int(row["image_id"])

            if image_id in seen_ids:
                raise ValueError(
                    f"Duplicate RGB image ID: {image_id}"
                )

            seen_ids.add(image_id)

            if image_id not in source_images:
                raise ValueError(
                    f"Image ID {image_id} not found in COCO JSON"
                )

            rgb_path = Path(row["rgb_path"])
            thermal_path = Path(row["thermal_path"])

            if not rgb_path.exists():
                raise FileNotFoundError(rgb_path)

            if not thermal_path.exists():
                raise FileNotFoundError(thermal_path)

            rgb = load_rgb(rgb_path, device)
            thermal = load_thermal(thermal_path, device)

            if rgb.shape[-2:] != thermal.shape[-2:]:
                raise ValueError(
                    f"Image pair {image_id} has different dimensions. "
                    "Register and align the sensors before fusion."
                )

            fused, gate = model(rgb, thermal)

            filename = f"fused_{image_id:08d}.png"

            save_path = image_dir / filename

            TF.to_pil_image(
                fused.squeeze(0).cpu()
            ).save(save_path)

            original = source_images[image_id]

            if (
                original["width"] != rgb.shape[-1]
                or original["height"] != rgb.shape[-2]
            ):
                raise ValueError(
                    f"COCO dimensions do not match RGB image {image_id}"
                )

            new_image = dict(original)
            new_image["file_name"] = f"images/{filename}"

            new_images.append(new_image)

            image_mapping[str(image_id)] = filename

            if index % 50 == 0:
                print(f"Processed {index}/{len(rows)} pairs")

    # Retain annotations only for images actually fused.
    new_annotations = [
        annotation
        for annotation in coco["annotations"]
        if annotation["image_id"] in seen_ids
    ]

    fused_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": new_images,
        "annotations": new_annotations,
        "categories": coco["categories"]
    }

    annotation_path = (
        output_dir / "_annotations.coco.json"
    )

    with open(
        annotation_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(fused_coco, file, indent=2)

    mapping_path = output_dir / "image_mapping.json"

    with open(
        mapping_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(image_mapping, file, indent=2)

    print("\nFused dataset created.")
    print("Images:", len(new_images))
    print("Annotations:", len(new_annotations))
    print("COCO JSON:", annotation_path)
    print("Image mapping:", mapping_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--pairs-csv", required=True)
    parser.add_argument("--coco-json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)

    main(parser.parse_args())
