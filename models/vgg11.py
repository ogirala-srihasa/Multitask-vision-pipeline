"""VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn


class VGG11Encoder(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns.
    """
    def _make_block(self, in_ch, out_ch, num_convs, batch_norm):
        layers = []
        for i in range(num_convs):
            layers.append(nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)


    def __init__(self, in_channels: int = 3, batch_norm: bool = True):
        super().__init__()
        self.block1 = self._make_block(in_channels, 64,  1, batch_norm)
        self.block2 = self._make_block(64,  128, 1, batch_norm)
        self.block3 = self._make_block(128, 256, 2, batch_norm)
        self.block4 = self._make_block(256, 512, 2, batch_norm)
        self.block5 = self._make_block(512, 512, 2, batch_norm)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, H, W].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor.
            - if return_features=True: (bottleneck, feature_dict).
        """
        h1 = self.block1(x)
        p1 = self.pool1(h1)
        h2 = self.block2(p1)
        p2 = self.pool2(h2)
        h3 = self.block3(p2)
        p3 = self.pool3(h3)
        h4 = self.block4(p3)
        p4 = self.pool4(h4)
        h5 = self.block5(p4)
        bottle_neck = self.pool5(h5)
        features = {
            "block1": h1,
            "block2": h2,
            "block3": h3,
            "block4": h4,
            "block5": h5, 
        }
        if return_features: return bottle_neck, features
        else: return bottle_neck

        
VGG11 = VGG11Encoder