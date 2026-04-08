"""Segmentation model
"""

import torch
import torch.nn as nn
from .vgg11 import VGG11
from .layers import CustomDropout
class VGG11UNet(nn.Module):
    """U-Net style segmentation network.
    """
    def _make_block(self, in_ch, out_ch, batch_norm,dropout_p,num_conv):
        layers = []
        layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        layers.append(CustomDropout(dropout_p))
        if(num_conv > 1):
            layers.append(nn.Conv2d(out_ch,out_ch,kernel_size= 3, padding= 1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            layers.append(CustomDropout(dropout_p))

        return nn.Sequential(*layers)
    

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5, batch_norm : bool = True):
        """
        Initialize the VGG11UNet model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the segmentation head.
        """
        super().__init__()
        # Encoder
        self.VGGhead = VGG11(in_channels= in_channels,batch_norm= batch_norm)

        # Decoder 
        self.up1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec1 = self._make_block(1024,512,batch_norm=batch_norm,dropout_p=dropout_p,num_conv= 2)
        self.up2 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec2 = self._make_block(1024,256,batch_norm,dropout_p,2)
        self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec3 = self._make_block(512,128,batch_norm,dropout_p,2)
        self.up4 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec4 = self._make_block(256,64,batch_norm,dropout_p,1)
        self.up5 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec5 = self._make_block(128,32,batch_norm,dropout_p,1)
        self.outConv = nn.Conv2d(32, num_classes, kernel_size=1)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        bottleneck, feature_maps = self.VGGhead(x, return_features=True)

        u1 = self.up1(bottleneck)
        cat1 = torch.cat([u1, feature_maps['block5']], dim=1)
        x1 = self.dec1(cat1)

        u2 = self.up2(x1)
        cat2 = torch.cat([u2, feature_maps['block4']], dim=1)
        x2 = self.dec2(cat2)

        u3 = self.up3(x2)
        cat3 = torch.cat([u3, feature_maps['block3']], dim=1)
        x3 = self.dec3(cat3)

        u4 = self.up4(x3)
        cat4 = torch.cat([u4, feature_maps['block2']], dim=1)
        x4 = self.dec4(cat4)

        u5 = self.up5(x4)
        cat5 = torch.cat([u5, feature_maps['block1']], dim=1)
        x5 = self.dec5(cat5)

        out = self.outConv(x5)

        return out