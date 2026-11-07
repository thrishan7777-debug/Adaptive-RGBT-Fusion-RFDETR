
import csv
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


CONDITIONS = [
    "normal",
    "mild_blur",
    "heavy_blur",
    "light_fog",
    "moderate_fog",
    "dense_fog"
]


def load_video_reports(results_dir):

    results_dir = Path(results_dir)

    report_files = list(
        results_dir.glob("Video_*/Video_*_metrics.csv")
    )

    if not report_files:
        raise FileNotFoundError(
            f"No video reports found in {results_dir}"
        )

    frames = []

    for report in report_files:

        data = pd.read_csv(report)

        required = {
            "video",
            "frame",
            "condition",
            "detection_count",
            "mean_detection_confidence"
        }

        if not required.issubset(data.columns):
            raise ValueError(
                f"Invalid report format: {report}"
            )

        frames.append(data)

    return pd.concat(frames, ignore_index=True)


def calculate_condition_summary(data):

    summary = []

    for condition in CONDITIONS:

        subset = data[
            data["condition"] == condition
        ]

        if subset.empty:
            continue

        confidence = pd.to_numeric(
            subset["mean_detection_confidence"],
            errors="coerce"
        )

        confidence_values = confidence.dropna()

        summary.append({
            "condition": condition,
            "total_frames": int(len(subset)),
            "total_detections": int(
                subset["detection_count"].sum()
            ),
            "mean_detections_per_frame": float(
                subset["detection_count"].mean()
            ),
            "mean_frame_detection_confidence": (
                float(confidence_values.mean())
                if not confidence_values.empty
                else None
            )
        })

    return pd.DataFrame(summary)


def calculate_video_summary(data):

    summary = []

    for video in sorted(data["video"].unique()):

        subset = data[data["video"] == video]

        for condition in CONDITIONS:

            condition_data = subset[
                subset["condition"] == condition
            ]

            if condition_data.empty:
                continue

            confidence = pd.to_numeric(
                condition_data[
                    "mean_detection_confidence"
                ],
                errors="coerce"
            ).dropna()

            summary.append({
                "video": int(video),
                "condition": condition,
                "frames": len(condition_data),
                "detections": int(
                    condition_data["detection_count"].sum()
                ),
                "mean_detections_per_frame": float(
                    condition_data["detection_count"].mean()
                ),
                "mean_frame_detection_confidence": (
                    float(confidence.mean())
                    if not confidence.empty else None
                )
            })

    return pd.DataFrame(summary)


def main(args):

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_video_reports(args.results)

    condition_summary = calculate_condition_summary(data)
    video_summary = calculate_video_summary(data)

    condition_summary.to_csv(
        output_dir / "condition_summary.csv",
        index=False
    )

    video_summary.to_csv(
        output_dir / "video_summary.csv",
        index=False
    )

    print("\nCondition summary:")
    print(condition_summary.to_string(index=False))

    print("\nVideo summary:")
    print(video_summary.to_string(index=False))

    print("\nReports saved to:", output_dir)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results",
        required=True
    )

    parser.add_argument(
        "--output",
        default="outputs/evaluation"
    )

    args = parser.parse_args()

    main(args)
