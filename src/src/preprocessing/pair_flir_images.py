
import json
from pathlib import Path
from collections import defaultdict


class FLIRPairValidator:
    """
    Validates RGB and thermal image pairs using FLIR metadata.

    Does not assume that RGB and thermal image filenames
    are identical.
    """

    def __init__(self, dataset_root):

        self.root = Path(dataset_root)

        self.rgb_json = (
            self.root / "video_rgb_test" / "coco.json"
        )

        self.thermal_json = (
            self.root / "video_thermal_test" / "coco.json"
        )

        self.mapping_json = (
            self.root / "rgb_to_thermal_vid_map.json"
        )

        self._check_files()

        self.rgb_data = self.load_json(self.rgb_json)
        self.thermal_data = self.load_json(
            self.thermal_json
        )

        self.mapping = self.load_json(
            self.mapping_json
        )

    @staticmethod
    def load_json(path):

        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _check_files(self):

        for path in [
            self.rgb_json,
            self.thermal_json,
            self.mapping_json
        ]:

            if not path.exists():
                raise FileNotFoundError(path)

    @staticmethod
    def get_video_id(image):

        return (
            image.get("extra_info", {}).get("video_id")
            or image.get("video_id")
        )

    def inspect_mapping(self):

        print("\nMAPPING STRUCTURE")
        print("-----------------")

        print("Mapping type:", type(self.mapping).__name__)

        if isinstance(self.mapping, dict):

            print("Number of mapping entries:", len(self.mapping))

            for index, (key, value) in enumerate(
                self.mapping.items()
            ):

                print("RGB video:", key)
                print("Thermal video:", value)

                if index == 4:
                    break

        elif isinstance(self.mapping, list):

            print("Number of entries:", len(self.mapping))

            for item in self.mapping[:5]:
                print(item)

        else:

            print("Unexpected mapping format.")

    def inspect_video_ids(self):

        rgb_ids = {
            self.get_video_id(image)
            for image in self.rgb_data.get("images", [])
        }

        thermal_ids = {
            self.get_video_id(image)
            for image in self.thermal_data.get("images", [])
        }

        print("\nVIDEO ID SUMMARY")
        print("----------------")

        print("RGB video IDs:", len(rgb_ids))
        print("Thermal video IDs:", len(thermal_ids))

        print("RGB IDs:", rgb_ids)
        print("Thermal IDs:", thermal_ids)

    def inspect_image_metadata(self):

        print("\nIMAGE METADATA")
        print("--------------")

        for modality, data in [
            ("RGB", self.rgb_data),
            ("THERMAL", self.thermal_data)
        ]:

            images = data.get("images", [])

            print(f"\n{modality}: {len(images)} images")

            for image in images[:3]:

                print({
                    "id": image.get("id"),
                    "file_name": image.get("file_name"),
                    "video_id": self.get_video_id(image),
                    "width": image.get("width"),
                    "height": image.get("height")
                })

    def run(self):

        self.inspect_mapping()
        self.inspect_video_ids()
        self.inspect_image_metadata()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        required=True
    )

    args = parser.parse_args()

    validator = FLIRPairValidator(args.dataset_root)

    validator.run()
