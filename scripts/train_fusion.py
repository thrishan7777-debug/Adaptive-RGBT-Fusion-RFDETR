
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF

from src.fusion.fusion_model import AdaptiveRGBThermalFusion
from src.fusion.fusion_losses import FusionLoss


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class PairedRGBThermalDataset(Dataset):
    """
    Expected dataset structure:

    dataset/
        rgb/
            image001.png
            image002.png
        thermal/
            image001.png
            image002.png

    RGB and thermal files must have matching relative paths.
    """

    def __init__(self, dataset_root, image_size=256):

        self.root = Path(dataset_root)
        self.rgb_root = self.root / "rgb"
        self.thermal_root = self.root / "thermal"
        self.image_size = image_size

        if not self.rgb_root.exists():
            raise FileNotFoundError(self.rgb_root)

        if not self.thermal_root.exists():
            raise FileNotFoundError(self.thermal_root)

        self.pairs = []

        for rgb_path in self.rgb_root.rglob("*"):

            if rgb_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            relative_path = rgb_path.relative_to(self.rgb_root)
            thermal_path = self.thermal_root / relative_path

            if thermal_path.exists():
                self.pairs.append((rgb_path, thermal_path))

        if not self.pairs:
            raise RuntimeError(
                "No matching RGB-thermal image pairs found."
            )

        print(f"Matched image pairs: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):

        rgb_path, thermal_path = self.pairs[index]

        rgb = Image.open(rgb_path).convert("RGB")
        thermal = Image.open(thermal_path).convert("L")

        size = (self.image_size, self.image_size)

        rgb = TF.resize(
            rgb,
            size,
            interpolation=TF.InterpolationMode.BILINEAR
        )

        thermal = TF.resize(
            thermal,
            size,
            interpolation=TF.InterpolationMode.BILINEAR
        )

        rgb = TF.to_tensor(rgb)
        thermal = TF.to_tensor(thermal)

        return rgb, thermal


def train(args):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Training device:", device)

    dataset = PairedRGBThermalDataset(
        args.dataset,
        image_size=args.image_size
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda")
    )

    model = AdaptiveRGBThermalFusion().to(device)

    criterion = FusionLoss().to(device)

    # Train the adaptive gate parameters.
    optimizer = torch.optim.Adam(
        model.gate.parameters(),
        lr=args.learning_rate
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):

        model.train()

        total_epoch_loss = 0.0

        for step, (rgb, thermal) in enumerate(loader):

            rgb = rgb.to(device)
            thermal = thermal.to(device)

            optimizer.zero_grad(set_to_none=True)

            fused, gate = model(rgb, thermal)

            rgb_y = model.rgb_to_luminance(rgb)

            fused_y = model.rgb_to_luminance(fused)

            thermal_y = model.thermal_to_intensity(thermal)

            losses = criterion(
                fused_y,
                rgb_y,
                thermal_y,
                gate
            )

            losses["total"].backward()

            optimizer.step()

            total_epoch_loss += losses["total"].item()

            if (step + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{args.epochs}] "
                    f"Step [{step + 1}/{len(loader)}] "
                    f"Loss: {losses['total'].item():.5f}"
                )

        average_loss = total_epoch_loss / len(loader)

        print(
            f"Epoch {epoch + 1}: "
            f"Average loss = {average_loss:.6f}"
        )

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "average_loss": average_loss,
            "image_size": args.image_size
        }

        torch.save(
            checkpoint,
            output_dir / "fusion_checkpoint_latest.pth"
        )

    print("Fusion training completed.")

    torch.save(
        model.state_dict(),
        output_dir / "adaptive_fusion_gate.pth"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=256
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0002
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/fusion"
    )

    args = parser.parse_args()

    train(args)

