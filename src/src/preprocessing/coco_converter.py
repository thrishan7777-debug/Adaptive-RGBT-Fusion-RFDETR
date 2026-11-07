
import json
from pathlib import Path
from collections import defaultdict


class COCOConverter:
    """
    Converts RGB COCO annotations into a new COCO dataset
    using an explicit mapping between original and fused images.

    Mapping format:
        {
            "1001": "fused_000001.png",
            "1002": "fused_000002.png"
        }

    Original RGB annotations are retained because fusion
    is assumed to preserve the RGB image geometry.
    """

    def __init__(self, source_json, output_dir):

        self.source_json = Path(source_json)
        self.output_dir = Path(output_dir)

        with open(
            self.source_json,
            "r",
            encoding="utf-8"
        ) as file:
            self.coco = json.load(file)

    def convert(self, image_mapping):

        images = []
        annotations = []

        annotation_id = 1

        source_images = {
            image["id"]: image
            for image in self.coco["images"]
        }

        for original_id, fused_filename in image_mapping.items():

            original_id = int(original_id)

            if original_id not in source_images:
                raise ValueError(
                    f"Unknown RGB image ID: {original_id}"
                )

            original = source_images[original_id]

            new_image = dict(original)

            new_image["file_name"] = fused_filename

            images.append(new_image)

            for annotation in self.coco["annotations"]:

                if annotation["image_id"] != original_id:
                    continue

                new_annotation = dict(annotation)

                new_annotation["id"] = annotation_id
                new_annotation["image_id"] = original_id

                annotations.append(new_annotation)

                annotation_id += 1

        converted = {
            "info": self.coco.get("info", {}),
            "licenses": self.coco.get("licenses", []),
            "images": images,
            "annotations": annotations,
            "categories": self.coco["categories"]
        }

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = (
            self.output_dir / "_annotations.coco.json"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                converted,
                file,
                indent=2
            )

        print("COCO conversion completed.")
        print("Images:", len(images))
        print("Annotations:", len(annotations))
        print("Categories:", len(converted["categories"]))
        print("Saved:", output_path)

        return output_path


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--source-json", required=True)
    parser.add_argument("--mapping-json", required=True)
    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    with open(
        args.mapping_json,
        "r",
        encoding="utf-8"
    ) as file:
        mapping = json.load(file)

    converter = COCOConverter(
        args.source_json,
        args.output_dir
    )

    converter.convert(mapping)
