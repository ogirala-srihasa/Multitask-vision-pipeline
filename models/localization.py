"""Localization modules
"""

import torch
import torch.nn as nn
from .layers import CustomDropout
from .vgg11 import VGG11

class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5, batch_norm: bool = True):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
        """
        super().__init__()
        # Convolution layers
        self.VGGhead = VGG11(in_channels= in_channels,batch_norm= batch_norm)

        # Fully connected layers
        self.layer1 = nn.Sequential(
            #Flattening
            nn.Flatten(),
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=1024, bias=True),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=4, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format in original image pixel space(not normalized values).
        """
        bottleneck = self.VGGhead(x)
        h1 = self.layer1(bottleneck)
        h2 = self.layer2(h1)
        box = self.layer3(h2) * 224.0
        return box