
import torch
import torch.nn as nn
import torch.nn.functional as F

from .adaptive_gate import AdaptiveFusionGate


class AdaptiveRGBThermalFusion(nn.Module):
    """
    Adaptive spatial RGB-thermal image fusion.

    RGB input:
        [B, 3, H, W], values in [0, 1]

    Thermal input:
        [B, 1, H, W] or [B, 3, H, W],
        values in [0, 1]

    Returns:
        Fused RGB image: [B, 3, H, W]
        Adaptive gate:   [B, 1, H, W]
    """

    def __init__(self, hidden_channels=32, contrast_kernel=7):
        super().__init__()

        if contrast_kernel < 1 or contrast_kernel % 2 == 0:
            raise ValueError(
                "contrast_kernel must be a positive odd number"
            )

        self.gate = AdaptiveFusionGate(
            in_channels=4,
            hidden_channels=hidden_channels
        )

        self.contrast_kernel = contrast_kernel

    @staticmethod
    def rgb_to_luminance(rgb):
        """Convert RGB channels into luminance."""

        weights = rgb.new_tensor(
            [0.299, 0.587, 0.114]
        ).view(1, 3, 1, 1)

        return (rgb * weights).sum(dim=1, keepdim=True)

    @staticmethod
    def thermal_to_intensity(thermal):
        """Convert thermal input into one intensity channel."""

        if thermal.shape[1] == 1:
            return thermal

        if thermal.shape[1] == 3:
            return thermal.mean(dim=1, keepdim=True)

        raise ValueError(
            "Thermal input must have 1 or 3 channels"
        )

    def local_contrast(self, image):
        """Calculate local contrast using average pooling."""

        padding = self.contrast_kernel // 2

        local_mean = F.avg_pool2d(
            image,
            kernel_size=self.contrast_kernel,
            stride=1,
            padding=padding
        )

        return torch.abs(image - local_mean)

    def forward(self, rgb, thermal, return_gate=True):

        if rgb.ndim != 4 or thermal.ndim != 4:
            raise ValueError(
                "Inputs must have shape [B, C, H, W]"
            )

        if rgb.shape[1] != 3:
            raise ValueError("RGB input must have 3 channels")

        if rgb.shape[0] != thermal.shape[0]:
            raise ValueError("Batch sizes must match")

        if rgb.shape[-2:] != thermal.shape[-2:]:
            raise ValueError(
                "RGB and thermal images must be spatially aligned"
            )

        if rgb.shape[-2] < self.contrast_kernel:
            raise ValueError("Image height is too small")

        if rgb.shape[-1] < self.contrast_kernel:
            raise ValueError("Image width is too small")

        # Step 1: Extract RGB luminance
        rgb_y = self.rgb_to_luminance(rgb)

        # Step 2: Extract thermal intensity
        thermal_y = self.thermal_to_intensity(thermal)

        # Step 3: Calculate local contrast
        rgb_contrast = self.local_contrast(rgb_y)
        thermal_contrast = self.local_contrast(thermal_y)

        # Step 4: Construct four-channel gate input
        gate_input = torch.cat(
            [
                rgb_y,
                thermal_y,
                rgb_contrast,
                thermal_contrast
            ],
            dim=1
        )

        # Step 5: Generate adaptive spatial gate
        gate = self.gate(gate_input)

        # Step 6: Weighted luminance fusion
        fused_y = (
            gate * rgb_y
            + (1.0 - gate) * thermal_y
        )

        # Step 7: Preserve RGB chromaticity
        eps = 1e-6

        fused_rgb = (
            rgb * (fused_y / (rgb_y + eps))
        ).clamp(0.0, 1.0)

        if return_gate:
            return fused_rgb, gate

        return fused_rgb


if __name__ == "__main__":

    model = AdaptiveRGBThermalFusion()

    rgb = torch.rand(2, 3, 256, 256)
    thermal = torch.rand(2, 1, 256, 256)

    fused, gate = model(rgb, thermal)

    print("RGB shape:", rgb.shape)
    print("Thermal shape:", thermal.shape)
    print("Fused shape:", fused.shape)
    print("Gate shape:", gate.shape)

    print("Fused minimum:", fused.min().item())
    print("Fused maximum:", fused.max().item())

    print("Gate minimum:", gate.min().item())
    print("Gate maximum:", gate.max().item())
