
import json
from pathlib import Path
from collections import defaultdict


class FLIRDatasetReader:
    """
    Reads FLIR ADAS v2 RGB and thermal COCO metadata.

    Expected structure:

    FLIR_ADAS_v2/
        video_rgb_test/
            data/
            coco.json

        video_thermal_test/
            data/
            coco.json

        rgb_to_thermal_vid_map.json

    RGB annotations are treated as the canonical
    object detection annotations.
    """

    def __init__(self, dataset_root):

        self.root = Path(dataset_root)

        self.rgb_dir = self.root / "video_rgb_test"
        self.thermal_dir = self.root / "video_thermal_test"

        self.rgb_json = self.rgb_dir / "coco.json"
        self.thermal_json = self.thermal_dir / "coco.json"

        self.mapping_json = (
            self.root / "rgb_to_thermal_vid_map.json"
        )

        self._check_files()

        self.rgb_coco = self._load_json(self.rgb_json)
        self.thermal_coco = self._load_json(
            self.thermal_json
        )

        self.video_mapping = self._load_json(
            self.mapping_json
        )

        self.rgb_images = self._index_images(
            self.rgb_coco
        )

        self.thermal_images = self._index_images(
            self.thermal_coco
        )

        self.categories = {
            item["id"]: item["name"]
            for item in self.rgb_coco.get(
                "categories", []
            )
        }

        self.annotations = self._index_annotations(
            self.rgb_coco
        )

    def _check_files(self):

        required = [
            self.rgb_json,
            self.thermal_json,
            self.mapping_json,
            self.rgb_dir / "data",
            self.thermal_dir / "data"
        ]

        missing = [
            str(path)
            for path in required
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Missing FLIR dataset paths:\n"
                + "\n".join(missing)
            )

    @staticmethod
    def _load_json(path):

        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _index_images(coco):

        return {
            image["id"]: image
            for image in coco.get("images", [])
        }

    @staticmethod
    def _index_annotations(coco):

        result = defaultdict(list)

        for annotation in coco.get(
            "annotations", []
        ):
            result[annotation["image_id"]].append(
                annotation
            )

        return result

    @staticmethod
    def _video_id(image):

        return (
            image.get("extra_info", {}).get("video_id")
            or image.get("video_id")
        )

    def get_category_summary(self):

        return {
            name: sum(
                1
                for annotation in self.rgb_coco.get(
                    "annotations", []
                )
                if annotation["category_id"] == category_id
            )
            for category_id, name in self.categories.items()
        }

    def get_rgb_annotation_records(self):

        records = []

        for image_id, image in self.rgb_images.items():

            records.append({
                "image_id": image_id,
                "file_name": image["file_name"],
                "video_id": self._video_id(image),
                "width": image.get("width"),
                "height": image.get("height"),
                "annotations": self.annotations.get(
                    image_id, []
                )
            })

        return records

    def inspect(self):

        print("RGB images:", len(self.rgb_images))
        print("Thermal images:", len(self.thermal_images))

        print(
            "RGB annotations:",
            len(self.rgb_coco.get("annotations", []))
        )

        print(
            "Thermal annotations:",
            len(self.thermal_coco.get("annotations", []))
        )

        print("\nRGB categories:")

        for category_id, name in self.categories.items():
            print(f"{category_id}: {name}")

        print("\nRGB category annotation counts:")

        for name, count in self.get_category_summary().items():
            print(f"{name}: {count}")

        print("\nMapping file type:", type(
            self.video_mapping
        ).__name__)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        required=True
    )

    args = parser.parse_args()

    dataset = FLIRDatasetReader(
        args.dataset_root
    )

    dataset.inspect()
