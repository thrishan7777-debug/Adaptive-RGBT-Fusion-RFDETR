
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from PIL import Image
import torchvision.transforms.functional as TF

from src.fusion.fusion_model import AdaptiveRGBThermalFusion
from src.detection.inference import RFDETRInference


CONDITIONS = {
    "normal": {"type": "normal"},
    "mild_blur": {"type": "blur", "kernel": 5},
    "heavy_blur": {"type": "blur", "kernel": 15},
    "light_fog": {"type": "fog", "strength": 0.15},
    "moderate_fog": {"type": "fog", "strength": 0.35},
    "dense_fog": {"type": "fog", "strength": 0.55},
}


def apply_degradation(rgb, condition):

    settings = CONDITIONS[condition]

    if settings["type"] == "normal":
        return rgb

    if settings["type"] == "blur":
        kernel = settings["kernel"]

        return cv2.GaussianBlur(
            rgb,
            (kernel, kernel),
            0
        )

    if settings["type"] == "fog":
        strength = settings["strength"]

        white = np.full_like(rgb, 255)

        return cv2.addWeighted(
            rgb,
            1.0 - strength,
            white,
            strength,
            0
        )

    raise ValueError(f"Unknown condition: {condition}")


def load_fusion_model(checkpoint, device):

    model = AdaptiveRGBThermalFusion().to(device)

    state = torch.load(
        checkpoint,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state)
    model.eval()

    return model


def fuse_frame(rgb_bgr, thermal_bgr, fusion_model, device):

    rgb = cv2.cvtColor(
        rgb_bgr,
        cv2.COLOR_BGR2RGB
    )

    thermal = cv2.cvtColor(
        thermal_bgr,
        cv2.COLOR_BGR2GRAY
    )

    if rgb.shape[:2] != thermal.shape[:2]:
        thermal = cv2.resize(
            thermal,
            (rgb.shape[1], rgb.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

    rgb_tensor = TF.to_tensor(
        Image.fromarray(rgb)
    ).unsqueeze(0).to(device)

    thermal_tensor = TF.to_tensor(
        Image.fromarray(thermal)
    ).unsqueeze(0).to(device)

    with torch.inference_mode():

        fused, gate = fusion_model(
            rgb_tensor,
            thermal_tensor
        )

    fused_image = (
        fused.squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    fused_image = np.clip(
        fused_image * 255,
        0,
        255
    ).astype(np.uint8)

    return fused_image


def process_video(
    video_number,
    rgb_path,
    thermal_path,
    fusion_model,
    detector,
    output_root,
    device
):

    rgb_cap = cv2.VideoCapture(str(rgb_path))
    thermal_cap = cv2.VideoCapture(str(thermal_path))

    if not rgb_cap.isOpened():
        raise RuntimeError(f"Cannot open RGB video: {rgb_path}")

    if not thermal_cap.isOpened():
        raise RuntimeError(
            f"Cannot open thermal video: {thermal_path}"
        )

    rgb_fps = rgb_cap.get(cv2.CAP_PROP_FPS)
    thermal_fps = thermal_cap.get(cv2.CAP_PROP_FPS)

    rgb_count = int(
        rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    thermal_count = int(
        thermal_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if rgb_fps <= 0 or thermal_fps <= 0:
        raise RuntimeError("Invalid video FPS.")

    print(f"\nVideo {video_number}")
    print("RGB FPS:", rgb_fps)
    print("Thermal FPS:", thermal_fps)
    print("RGB frames:", rgb_count)
    print("Thermal frames:", thermal_count)

    if not np.isclose(rgb_fps, thermal_fps, rtol=0.01):
        print(
            "WARNING: RGB and thermal FPS differ. "
            "Frame-by-frame pairing may not be synchronized."
        )

    if rgb_count != thermal_count:
        print(
            "WARNING: Frame counts differ. "
            "Processing the shorter stream."
        )

    fps = rgb_fps
    total_frames = min(rgb_count, thermal_count)

    if total_frames <= 0:
        raise RuntimeError("No video frames available.")

    output_dir = output_root / f"Video_{video_number}"
    output_dir.mkdir(parents=True, exist_ok=True)

    writers = {}

    for condition in CONDITIONS:

        output_path = (
            output_dir /
            f"Video_{video_number}_{condition}.mp4"
        )

        writers[condition] = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (
                int(rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
        )

        if not writers[condition].isOpened():
            raise RuntimeError(
                f"Cannot create output video: {output_path}"
            )

    report_rows = []

    frame_index = 0

    try:

        while frame_index < total_frames:

            rgb_ok, rgb_frame = rgb_cap.read()
            thermal_ok, thermal_frame = thermal_cap.read()

            if not rgb_ok or not thermal_ok:
                break

            fused_rgb = fuse_frame(
                rgb_frame,
                thermal_frame,
                fusion_model,
                device
            )

            frame_results = {}

            for condition in CONDITIONS:

                degraded_rgb = apply_degradation(
                    fused_rgb.copy(),
                    condition
                )

                pil_image = Image.fromarray(degraded_rgb)

                predictions = detector.model.predict(
                    pil_image,
                    threshold=detector.confidence
                )

                annotated, records = detector.annotate(
                    pil_image,
                    predictions,
                    [
                        "person",
                        "bike",
                        "car",
                        "motor",
                        "truck",
                        "light",
                        "hydrant",
                        "sign",
                        "other vehicle"
                    ]
                )

                output_bgr = cv2.cvtColor(
                    np.asarray(annotated),
                    cv2.COLOR_RGB2BGR
                )

                writers[condition].write(output_bgr)

                confidences = [
                    record["confidence"]
                    for record in records
                ]

                frame_results[condition] = {
                    "detections": len(records),
                    "mean_confidence": (
                        float(np.mean(confidences))
                        if confidences else None
                    )
                }

                report_rows.append({
                    "video": video_number,
                    "frame": frame_index,
                    "condition": condition,
                    "detection_count": len(records),
                    "mean_detection_confidence": (
                        float(np.mean(confidences))
                        if confidences else ""
                    )
                })

            if frame_index % 100 == 0:
                print(
                    f"Video {video_number}: "
                    f"{frame_index}/{total_frames} frames"
                )

            frame_index += 1

    finally:

        rgb_cap.release()
        thermal_cap.release()

        for writer in writers.values():
            writer.release()

    report_path = (
        output_dir / f"Video_{video_number}_metrics.csv"
    )

    with open(
        report_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "video",
            "frame",
            "condition",
            "detection_count",
            "mean_detection_confidence"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(report_rows)

    print(
        f"Completed Video {video_number}: "
        f"{frame_index} frames"
    )


def main(args):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    fusion_model = load_fusion_model(
        args.fusion_checkpoint,
        device
    )

    detector = RFDETRInference(
        args.detector_checkpoint,
        confidence=args.confidence
    )

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    for number in range(1, 6):

        rgb_path = (
            Path(args.video_dir) /
            f"Video_{number}_RGB.mp4"
        )

        thermal_path = (
            Path(args.video_dir) /
            f"Video_{number}_Thermal.mp4"
        )

        if not rgb_path.exists():
            raise FileNotFoundError(rgb_path)

        if not thermal_path.exists():
            raise FileNotFoundError(thermal_path)

        process_video(
            number,
            rgb_path,
            thermal_path,
            fusion_model,
            detector,
            output_root,
            device
        )

    print("\nAll five video pairs processed.")
    print("Expected outputs: up to 30 annotated videos.")
    print("Output directory:", output_root)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video-dir",
        required=True
    )

    parser.add_argument(
        "--fusion-checkpoint",
        required=True
    )

    parser.add_argument(
        "--detector-checkpoint",
        required=True
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35
    )

    parser.add_argument(
        "--output",
        default="outputs/videos"
    )

    main(parser.parse_args())
