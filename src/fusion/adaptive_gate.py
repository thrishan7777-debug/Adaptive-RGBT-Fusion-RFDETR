
import torch
import torch.nn as nn


class AdaptiveFusionGate(nn.Module):
    """
    CNN-based spatial gate for adaptive RGB-thermal fusion.

    Input channels:
        0: RGB luminance
        1: Thermal intensity
        2: RGB local contrast
        3: Thermal local contrast

    Output:
        Spatial gate with values between 0 and 1.
    """

    def __init__(self, in_channels=4, hidden_channels=32):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_channels,
                hidden_channels // 2,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_channels // 2,
                1,
                kernel_size=1
            ),

            nn.Sigmoid()
        )

    def forward(self, features):
        """
        Args:
            features: Tensor of shape [B, 4, H, W]

        Returns:
            gate: Tensor of shape [B, 1, H, W]
        """

        if features.ndim != 4:
            raise ValueError(
                "Expected input shape [B, 4, H, W]"
            )

        if features.shape[1] != 4:
            raise ValueError(
                "Adaptive gate requires exactly 4 input channels"
            )

        return self.network(features)


if __name__ == "__main__":
    gate_model = AdaptiveFusionGate()

    sample = torch.rand(2, 4, 256, 256)

    gate = gate_model(sample)

    print("Input shape:", sample.shape)
    print("Gate shape:", gate.shape)
    print("Gate minimum:", gate.min().item())
    print("Gate maximum:", gate.max().item())
