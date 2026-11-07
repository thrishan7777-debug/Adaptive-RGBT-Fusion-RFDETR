
import argparse
import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw

from rfdetr import RFDETRLarge


class RFDETRInference:

    def __init__(self, checkpoint, confidence=0.35):

        self.device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.confidence = confidence

        checkpoint = Path(checkpoint)

        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

        self.model = RFDETRLarge(
            pretrain_weights=str(checkpoint)
        )

        self.model.optimize_for_inference()

        print("RF-DETR-L loaded.")
        print("Device:", self.device)

    def predict(self, image_path):

        image = Image.open(image_path).convert("RGB")

        predictions = self.model.predict(
            image,
            threshold=self.confidence
        )

        return image, predictions

    @staticmethod
    def get_prediction_data(predictions):

        boxes = np.asarray(predictions.xyxy)
        scores = np.asarray(predictions.confidence)
        class_ids = np.asarray(predictions.class_id)

        return boxes, scores, class_ids

    def annotate(self, image, predictions, class_names):

        boxes, scores, class_ids = (
            self.get_prediction_data(predictions)
        )

        output = image.copy()
        draw = ImageDraw.Draw(output)

        records = []

        for box, score, class_id in zip(
            boxes, scores, class_ids
        ):

            x1, y1, x2, y2 = map(float, box)

            class_id = int(class_id)

            if 0 <= class_id < len(class_names):
                label = class_names[class_id]
            else:
                label = f"class_{class_id}"

            draw.rectangle(
                [x1, y1, x2, y2],
                outline="lime",
                width=3
            )

            draw.text(
                (x1, max(0, y1 - 18)),
                f"{label}: {float(score):.2f}",
                fill="yellow"
            )

            records.append({
                "class_id": class_id,
                "class_name": label,
                "confidence": float(score),
                "bbox_xyxy": [
                    x1, y1, x2, y2
                ]
            })

        return output, records


def main(args):

    detector = RFDETRInference(
        checkpoint=args.checkpoint,
        confidence=args.confidence
    )

    with open(
        args.categories,
        "r",
        encoding="utf-8"
    ) as file:
        class_names = json.load(file)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image, predictions = detector.predict(args.image)

    annotated, records = detector.annotate(
        image,
        predictions,
        class_names
    )

    annotated.save(output_dir / "detections.png")

    with open(
        output_dir / "detections.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(records, file, indent=2)

    print("\nInference completed.")
    print("Detections:", len(records))
    print("Saved:", output_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--categories", required=True)

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35
    )

    parser.add_argument(
        "--output",
        default="outputs/detections"
    )

    main(parser.parse_args())
