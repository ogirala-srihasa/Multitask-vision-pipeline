"""Unified multi-task model
"""
import os
import torch
import torch.nn as nn
from .vgg11 import VGG11
from .layers import CustomDropout

class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""
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

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, classifier_path: str = "classifier.pth", localizer_path: str = "localizer.pth", unet_path: str = "unet.pth",batch_norm: bool = True, dropout_p = 0.5, encoder_backbone = 'classifier'):
        """
        Initialize the shared backbone/heads using these trained weights.
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
        """
        super().__init__()
        import gdown
        # ---- set 1 -----
        # gdown.download(id="1n4q73AV9ijs5d_T3Qayrr1sGTVCbbi2p", output=classifier_path, quiet=False)
        # gdown.download(id="1FoIOcD2UUzm8JzCoj-ZuhvQbvfRmaKyM", output=localizer_path, quiet=False)
        # gdown.download(id="1SigUIOub_XJ6k0dsO_i_1zorf0LA_Yq4", output=unet_path, quiet=False)

        # ----- set 2 -----
        # gdown.download(id="1UlHUdhT65SM6Q-gnJW1U_aUvok5SfxqY", output=classifier_path, quiet=False)
        # gdown.download(id="1KJgHl-nqLaQZL2ZDtUnRbAqyGQ_ZxCoS", output=localizer_path, quiet=False)
        # gdown.download(id="1EXLixl-KhX7hitpn7APVuu82b3k_8Pj2", output=unet_path, quiet=False)

        # ---- set 3 ----
        # gdown.download(id="1MHl6nIjHZroGB79A0oxjqp1vZML5j6Ls", output=classifier_path, quiet=False)
        # gdown.download(id="13F0Ykb3qEp0Br_Nps0R_6NReNhjAuOsj", output=localizer_path, quiet=False)
        # gdown.download(id="1oJcxfEdpr-ppXXmHrYwhPjMMtb-G2Sko", output=unet_path, quiet=False)

        # --- set 4 ----
        gdown.download(id="1n4q73AV9ijs5d_T3Qayrr1sGTVCbbi2p", output=classifier_path, quiet=False)
        gdown.download(id="1IryZXscys6zUgVXGKIflvw-MekpE-6dH", output=localizer_path, quiet=False)
        gdown.download(id="lqNpDYWZjhtwaaAHyLhyUInGQsd0l7dS", output=unet_path, quiet=False)


        
        

        #shared Encoder
        self.VGGhead = VGG11(in_channels= in_channels, batch_norm= batch_norm)

        #classification head
        self.classification_layer1 = nn.Sequential(
            #Flattening
            nn.Flatten(),
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p)
        )
        self.classification_layer2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p)
        )
        self.classification_layer3 = nn.Linear(in_features=4096, out_features=num_breeds, bias=True)


        #Localization head
        self.localization_layer1 = nn.Sequential(
            #Flattening
            nn.Flatten(),
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p)
        )
        self.localization_layer2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=1024, bias=True),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p)
        )
        self.localization_layer3 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=4, bias=True),
            nn.Sigmoid()
        )

        #Segmentation head
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
        self.outConv = nn.Conv2d(32, seg_classes, kernel_size=1)

        #Loading weights
        if os.path.exists(unet_path) and os.path.exists(classifier_path) and os.path.exists(localizer_path):
            unet_checkpoint = torch.load(unet_path, map_location="cpu")
            class_checkpoint = torch.load(classifier_path, map_location="cpu")
            loc_checkpoint = torch.load(localizer_path, map_location="cpu")

            unet_state = unet_checkpoint.get("state_dict", unet_checkpoint)
            class_state = class_checkpoint.get("state_dict", class_checkpoint)
            loc_state = loc_checkpoint.get("state_dict", loc_checkpoint)

            
            classification_head = {k.replace("layer", "classification_layer"): v for k, v in class_state.items() if not k.startswith("VGGhead.")}
            localization_head = {k.replace("layer", "localization_layer"): v for k, v in loc_state.items() if not k.startswith("VGGhead.")}
            unet_head = {k: v for k, v in unet_state.items() if not k.startswith("VGGhead.")}

            classification_encoder = {k.replace("VGGhead.", ""): v for k, v in class_state.items() if k.startswith("VGGhead.")}
            localization_encoder = {k.replace("VGGhead.", ""): v for k, v in loc_state.items() if k.startswith("VGGhead.")}
            unet_encoder = {k.replace("VGGhead.", ""): v for k, v in unet_state.items() if k.startswith("VGGhead.")}

            # Loading the encoder
            if encoder_backbone == "unet":
                self.VGGhead.load_state_dict(unet_encoder, strict=False)
            elif encoder_backbone == "classifier":
                self.VGGhead.load_state_dict(classification_encoder, strict=False)
            elif encoder_backbone == "localizer":
                self.VGGhead.load_state_dict(localization_encoder, strict=False)
            else:
                raise ValueError("Invalid Backbone to extract Encoder from")
            
            # Loading the heads
            self.load_state_dict(classification_head, strict=False)
            self.load_state_dict(localization_head, strict=False)
            self.load_state_dict(unet_head, strict=False)
        else:
            print("[WARNING] One or more checkpoints missing. Initializing Multi-task model with random weights for architecture verification.")


    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        #obataining encoded data
        bottleneck, feature_maps = self.VGGhead(x, return_features=True)

        #classification
        h1 = self.classification_layer1(bottleneck)
        h2 = self.classification_layer2(h1)
        logits = self.classification_layer3(h2)

        #localization
        h1 = self.localization_layer1(bottleneck)
        h2 = self.localization_layer2(h1)
        box = self.localization_layer3(h2) * 224.0

        #segmentation
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

        return {
            "classification": logits,
            "localization": box,
            "segmentation": out
        }

