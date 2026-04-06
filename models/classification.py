"""Classification components
"""

import torch
import torch.nn as nn
from .layers import CustomDropout
from .vgg11 import VGG11


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5, batch_norm: bool = True):
        """
        Initialize the VGG11Classifier model.
        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the classifier head.
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
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p)
        )
        self.layer3 = nn.Linear(in_features=4096, out_features=num_classes, bias=True)
        



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            Classification logits [B, num_classes].
        """
        bottleneck = self.VGGhead(x)
        h1 = self.layer1(bottleneck)
        h2 = self.layer2(h1)
        logits = self.layer3(h2)
        return logits