
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


CONDITION_ORDER = [
    "normal",
    "mild_blur",
    "heavy_blur",
    "light_fog",
    "moderate_fog",
    "dense_fog"
]


def load_summary(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    data = pd.read_csv(path)

    required = {
        "condition",
        "total_detections",
        "mean_detections_per_frame",
        "mean_frame_detection_confidence"
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    return data


def order_conditions(data):

    data = data.copy()

    data["condition"] = pd.Categorical(
        data["condition"],
        categories=CONDITION_ORDER,
        ordered=True
    )

    return data.sort_values("condition")


def plot_detections(data, output_dir):

    plt.figure(figsize=(10, 6))

    plt.bar(
        data["condition"].astype(str),
        data["total_detections"]
    )

    plt.title("Total Object Detections by Condition")
    plt.xlabel("Environmental Condition")
    plt.ylabel("Total Detections")

    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    plt.savefig(
        output_dir / "detection_comparison.png",
        dpi=300
    )

    plt.close()


def plot_mean_detections(data, output_dir):

    plt.figure(figsize=(10, 6))

    plt.plot(
        data["condition"].astype(str),
        data["mean_detections_per_frame"],
        marker="o"
    )

    plt.title("Mean Detections per Frame")
    plt.xlabel("Environmental Condition")
    plt.ylabel("Mean Detections")

    plt.xticks(rotation=25, ha="right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "mean_detections.png",
        dpi=300
    )

    plt.close()


def plot_confidence(data, output_dir):

    plt.figure(figsize=(10, 6))

    plt.plot(
        data["condition"].astype(str),
        data["mean_frame_detection_confidence"],
        marker="o"
    )

    plt.title("Mean Frame Detection Confidence")
    plt.xlabel("Environmental Condition")
    plt.ylabel("Mean Confidence")

    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "confidence_comparison.png",
        dpi=300
    )

    plt.close()


def plot_video_comparison(video_data, output_dir):

    plt.figure(figsize=(11, 6))

    for video in sorted(video_data["video"].unique()):

        subset = video_data[
            video_data["video"] == video
        ].copy()

        subset["condition"] = pd.Categorical(
            subset["condition"],
            categories=CONDITION_ORDER,
            ordered=True
        )

        subset = subset.sort_values("condition")

        plt.plot(
            subset["condition"].astype(str),
            subset["mean_detections_per_frame"],
            marker="o",
            label=f"Video {video}"
        )

    plt.title("Per-Video Detection Comparison")
    plt.xlabel("Environmental Condition")
    plt.ylabel("Mean Detections per Frame")

    plt.xticks(rotation=25, ha="right")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "per_video_comparison.png",
        dpi=300
    )

    plt.close()


def main(args):

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    condition_data = order_conditions(
        load_summary(args.condition_summary)
    )

    video_data = pd.read_csv(
        args.video_summary
    )

    plot_detections(condition_data, output_dir)

    plot_mean_detections(condition_data, output_dir)

    plot_confidence(condition_data, output_dir)

    plot_video_comparison(video_data, output_dir)

    print("\nVisualization completed.")

    for file in sorted(output_dir.glob("*.png")):
        print("Saved:", file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--condition-summary",
        required=True
    )

    parser.add_argument(
        "--video-summary",
        required=True
    )

    parser.add_argument(
        "--output",
        default="outputs/figures"
    )

    main(parser.parse_args())
