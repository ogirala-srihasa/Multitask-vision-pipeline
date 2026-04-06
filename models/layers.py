"""Reusable custom layers 
"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Custom Dropout layer.
    """

    def __init__(self, p: float = 0.5):
        """
        Initialize the CustomDropout layer.

        Args:
            p: Dropout probability.
        """
        super().__init__()
        if p < 0.0 or p >= 1.0:
            raise ValueError(f"Dropout probability must be in the range [0, 1], but got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CustomDropout layer.

        Args:
            x: Input tensor for shape [B, C, H, W].

        Returns:
            Output tensor.
        """
        #if the layer is called during eval mode
        if not self.training:
            return x
        #creating the mask
        mask = torch.empty_like(x).bernoulli_(1 - self.p)
        #applying dropout
        y = x * mask
        #inverted dropout
        y = y/((1-self.p))
        return y
