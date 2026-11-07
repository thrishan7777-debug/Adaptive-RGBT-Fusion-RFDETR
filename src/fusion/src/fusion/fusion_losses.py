
import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionLoss(nn.Module):
    """
    Self-supervised loss for adaptive RGB-thermal fusion.

    Components:
        1. Gradient preservation loss
        2. Intensity preservation loss
        3. Gate entropy regularization
    """

    def __init__(
        self,
        intensity_weight=0.25,
        entropy_weight=0.01
    ):
        super().__init__()

        self.intensity_weight = intensity_weight
        self.entropy_weight = entropy_weight

        # Sobel filters for edge extraction
        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def gradient(self, image):
        """Extract horizontal and vertical image gradients."""

        gx = F.conv2d(
            image,
            self.sobel_x,
            padding=1
        )

        gy = F.conv2d(
            image,
            self.sobel_y,
            padding=1
        )

        magnitude = torch.sqrt(
            gx.pow(2) + gy.pow(2) + 1e-8
        )

        return magnitude

    def intensity_loss(self, fused, rgb, thermal):
        """
        Encourage fused intensity to preserve the stronger
        intensity response from either modality.
        """

        target = torch.maximum(rgb, thermal)

        return F.l1_loss(fused, target)

    def gradient_loss(self, fused, rgb, thermal):
        """
        Preserve strong edge information from RGB
        and thermal modalities.
        """

        fused_gradient = self.gradient(fused)
        rgb_gradient = self.gradient(rgb)
        thermal_gradient = self.gradient(thermal)

        target_gradient = torch.maximum(
            rgb_gradient,
            thermal_gradient
        )

        return F.l1_loss(
            fused_gradient,
            target_gradient
        )

    @staticmethod
    def gate_entropy(gate):
        """
        Calculate mean binary entropy of the gate.

        Higher entropy means the gate is less extreme.
        """

        eps = 1e-7

        gate = gate.clamp(eps, 1.0 - eps)

        entropy = -(
            gate * torch.log(gate)
            + (1.0 - gate) * torch.log(1.0 - gate)
        )

        return entropy.mean()

    def forward(
        self,
        fused,
        rgb_luminance,
        thermal_intensity,
        gate
    ):

        loss_intensity = self.intensity_loss(
            fused,
            rgb_luminance,
            thermal_intensity
        )

        loss_gradient = self.gradient_loss(
            fused,
            rgb_luminance,
            thermal_intensity
        )

        entropy = self.gate_entropy(gate)

        total_loss = (
            loss_gradient
            + self.intensity_weight * loss_intensity
            - self.entropy_weight * entropy
        )

        return {
            "total": total_loss,
            "gradient": loss_gradient,
            "intensity": loss_intensity,
            "entropy": entropy
        }


if __name__ == "__main__":

    criterion = FusionLoss()

    rgb = torch.rand(2, 1, 128, 128)
    thermal = torch.rand(2, 1, 128, 128)

    fused = torch.rand(
        2, 1, 128, 128,
        requires_grad=True
    )

    gate = torch.rand(
        2, 1, 128, 128,
        requires_grad=True
    )

    losses = criterion(
        fused,
        rgb,
        thermal,
        gate
    )

    print("Fusion loss results:")

    for name, value in losses.items():
        print(f"{name}: {value.item():.6f}")

    losses["total"].backward()

    print("Backward propagation successful.")
