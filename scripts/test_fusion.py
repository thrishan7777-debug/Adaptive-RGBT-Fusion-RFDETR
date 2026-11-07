
import argparse
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms.functional as TF

from src.fusion.fusion_model import AdaptiveRGBThermalFusion


def load_image(path, channels, image_size, device):
    """Load and normalize an RGB or thermal image."""

    image = Image.open(path)

    if channels == 3:
        image = image.convert("RGB")
    else:
        image = image.convert("L")

    image = TF.resize(
        image,
        (image_size, image_size),
        interpolation=TF.InterpolationMode.BILINEAR
    )

    tensor = TF.to_tensor(image).unsqueeze(0)

    return tensor.to(device)


def save_tensor(tensor, output_path):
    """Save a normalized image tensor."""

    tensor = tensor.detach().cpu().squeeze(0).clamp(0, 1)

    TF.to_pil_image(tensor).save(output_path)


def main(args):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    model = AdaptiveRGBThermalFusion().to(device)

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model.eval()

    rgb = load_image(
        args.rgb,
        channels=3,
        image_size=args.image_size,
        device=device
    )

    thermal = load_image(
        args.thermal,
        channels=1,
        image_size=args.image_size,
        device=device
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():

        fused, gate = model(rgb, thermal)

    save_tensor(rgb, output_dir / "rgb_input.png")
    save_tensor(thermal, output_dir / "thermal_input.png")
    save_tensor(fused, output_dir / "fused_output.png")
    save_tensor(gate, output_dir / "adaptive_gate.png")

    print("\nFusion testing completed.")

    print("RGB shape:", tuple(rgb.shape))
    print("Thermal shape:", tuple(thermal.shape))
    print("Fused shape:", tuple(fused.shape))
    print("Gate shape:", tuple(gate.shape))

    print("Gate minimum:", gate.min().item())
    print("Gate maximum:", gate.max().item())

    print("Saved results to:", output_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Test adaptive RGB-thermal fusion"
    )

    parser.add_argument("--rgb", required=True)
    parser.add_argument("--thermal", required=True)

    parser.add_argument(
        "--checkpoint",
        default="outputs/fusion/adaptive_fusion_gate.pth"
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=256
    )

    parser.add_argument(
        "--output",
        default="outputs/fusion_test"
    )

    args = parser.parse_args()

    main(args)
