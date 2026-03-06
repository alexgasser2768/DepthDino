import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import os, timm, logging

logger = logging.getLogger(__name__)


PREPROCESS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# From MobileNet paper
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, upscale_factor=1):
        super().__init__()

        # 1. Depthwise Convolution
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False  # Batch norm handles the bias
        )
        self.bn_dw = nn.BatchNorm2d(in_channels)
        self.act_dw = nn.ReLU(inplace=True)

        # If upscale=1, multiplier is 1 (normal pointwise).
        # If upscale=2, multiplier is 4 (expanded pointwise for upsampling).
        expansion_multiplier = upscale_factor ** 2
        pw_out_channels = out_channels * expansion_multiplier

        # 2. Pointwise Convolution (Handles dynamic expansion)
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=pw_out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False  # Batchnorm handles the bias
        )

        # 3. Pixel Shuffle
        # Acts as an identity pass-through if upscale_factor == 1
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        # 4. Final Normalization and Activation
        self.bn_pw = nn.BatchNorm2d(out_channels)
        self.act_pw = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn_dw(x)
        x = self.act_dw(x)
        
        x = self.pointwise(x)
        x = self.pixel_shuffle(x) 
        x = self.bn_pw(x)
        x = self.act_pw(x)
        
        return x


class ConvNeXtDepthModel(nn.Module):
    def __init__(self, arch='convnext_tiny.dinov3_lvd1689m', mlp_weights_path=None, pretrained=True):
        super().__init__()

        # --- B. Instantiate Backbone (Frozen) ---
        logger.info(f"Loading Backbone from timm: {arch}")
        # global_pool='' ensures we get the 7x7 spatial feature map, not a vector
        self.backbone = timm.create_model(arch, pretrained=pretrained, features_only=True)

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # --- C. Learnable MLP Decoder ---
        enc_channels = self.backbone.feature_info.channels()
        self.total_concat_channels = sum(enc_channels)  # 96 + 192 + 384 + 768 = 1440

        # Stage 1: 1/4 -> 1/2
        # Input: feat4 + feat8 + feat16 + feat32 (total_concat_channels)
        self.up1 = DepthwiseSeparableConv(self.total_concat_channels, 256, upscale_factor=2)

        # Stage 2: 1/2 -> 1/1 (Original Resolution)
        self.up2 = DepthwiseSeparableConv(256, 128, upscale_factor=2)

        # Final Projection to Depth (1 channel)
        self.head = nn.Sequential(
            DepthwiseSeparableConv(128, 64),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        if mlp_weights_path is not None and os.path.exists(mlp_weights_path):
            state_dict = torch.load(mlp_weights_path, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']

            # Load everything (Backbone + Decoder)
            missing, _ = self.load_state_dict(state_dict, strict=False)

            if len(missing) == 0:
                logging.info("Success: Full model (Backbone + MLP) loaded.")
            else:
                logging.warning(f"Partial load. Missing keys: {len(missing)}")

    def forward(self, x):
        # 1. Extract Features (Frozen)
        # Returns a list: [ f4,    f8,    f16,    f32]
        # Strides:         1/4    1/8    1/16    1/32
        features = self.backbone(x)
        f4, f8, f16, f32 = features
        target_size = f4.shape[-2:] # 1/4 resolution

        # 2. Bilinear Upsample all to 1/4 resolution
        f8_up  = F.interpolate(f8,  size=target_size, mode='bilinear', align_corners=False)
        f16_up = F.interpolate(f16, size=target_size, mode='bilinear', align_corners=False)
        f32_up = F.interpolate(f32, size=target_size, mode='bilinear', align_corners=False)

        # 3. Stack (Concatenate)
        # Resulting shape: [B, sum(C), H/4, W/4]
        merged = torch.cat([f4, f8_up, f16_up, f32_up], dim=1)

        # Final prediction
        x = self.up1(merged)      # [B, 256, H/2, W/2]
        x = self.up2(x)           # [B, 128,   H,   W]
        depth_map = self.head(x)  # [B,   1,   H,   W]

        return depth_map
