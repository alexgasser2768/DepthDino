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
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # 1. Depthwise Convolution: Spatial Filtering
        # By setting groups=in_channels, PyTorch applies exactly one 
        # spatial filter to each input channel individually.
        self.depthwise = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=in_channels, 
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False           # No bias needed before BatchNorm
        )
        self.bn_dw = nn.BatchNorm2d(in_channels)
        self.act_dw = nn.ReLU(inplace=True)

        # 2. Pointwise Convolution: Channel Mixing
        # A standard 1x1 convolution to linearly combine the channels.
        self.pointwise = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=False
        )
        self.bn_pw = nn.BatchNorm2d(out_channels)
        self.act_pw = nn.ReLU(inplace=True)

    def forward(self, x):
        # Apply depthwise spatial filtering
        x = self.depthwise(x)
        x = self.bn_dw(x)
        x = self.act_dw(x)

        # Apply pointwise channel mixing
        x = self.pointwise(x)
        x = self.bn_pw(x)
        x = self.act_pw(x)
        
        return x


class LearnableUpsampleBlock(DepthwiseSeparableConv):
    def __init__(self, in_channels, out_channels):
        # To upsample by 2x, we need 4x the channels (2*2).
        super().__init__(in_channels, out_channels * 4, stride=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)  # PixelShuffle packs channels into space.

    def forward(self, x):
        # Step 1: Spatial filtering with intermediate stabilization
        x = self.depthwise(x)
        x = self.bn_dw(x)
        x = self.act_dw(x)

        # Step 2: Channel mixing and expansion for upsampling
        x = self.pointwise(x)

        # Step 3: Shift channels into spatial dimensions
        x = self.pixel_shuffle(x)  # [B, C*4, H, W] -> [B, C, H*2, W*2]

        # Step 4: Final normalization and activation on the upsampled features
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
        self.up1 = LearnableUpsampleBlock(self.total_concat_channels, 512)
        
        # Stage 2: 1/2 -> 1/1 (Original Resolution)
        # Input: up1 output (512)
        self.up2 = LearnableUpsampleBlock(512, 256)

        # Final Projection to Depth (1 channel)
        self.head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Softplus() # Force positive depth
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
        x = self.up1(merged)      # [B, 512, H/2, W/2]
        x = self.up2(x)           # [B, 256,   H,   W]
        depth_map = self.head(x)  # [B,   1,   H,   W]

        return depth_map
