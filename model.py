import torch
import torch.nn as nn
from torchvision import transforms

import os, json, timm, logging
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


PREPROCESS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


class LearnableUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # PixelShuffle packs channels into space. 
        # To upsample by 2x, we need 4x the channels (2*2).
        self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.pixel_shuffle(x) # [B, C*4, H, W] -> [B, C, H*2, W*2]

        # Apply LayerNorm (channels last)
        x = x.permute(0, 2, 3, 1) # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2) # [N, H, W, C] -> [N, C, H, W]

        return x


class ConvNeXtDepthModel(nn.Module):
    def __init__(self, config_path, dino_weights_path, mlp_weights_path = None):
        super().__init__()

        # --- A. Parse Config ---
        with open(config_path, 'r') as f:
            cfg = json.load(f)

        arch = cfg.get("architecture", "convnext_tiny")
        num_features = cfg.get("num_features", 768)

        # --- B. Instantiate Backbone (Frozen) ---
        logger.info(f"Loading Backbone: {arch}")
        # global_pool='' ensures we get the 7x7 spatial feature map, not a vector
        self.backbone = timm.create_model(arch, pretrained=False, num_classes=0, global_pool='')

        # Load Safetensors and freeze backbone
        logger.info(f"Loading weights from {dino_weights_path}...")
        raw_weights = load_file(dino_weights_path)
        self.backbone.load_state_dict(raw_weights, strict=True)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # --- C. Learnable MLP Decoder ---
        # ConvNeXt output is 1/32 scale (e.g. 7x7 for 224 input).
        # We need 5 upsampling stages to get back to 1/1 scale (2^5 = 32).
        self.decoder = nn.Sequential(
            # Stage 1: 1/32 -> 1/16
            LearnableUpsampleBlock(num_features, 512),
            # Stage 2: 1/16 -> 1/8
            LearnableUpsampleBlock(512, 256),
            # Stage 3: 1/8 -> 1/4
            LearnableUpsampleBlock(256, 128),
            # Stage 4: 1/4 -> 1/2
            LearnableUpsampleBlock(128, 64),
            # Stage 5: 1/2 -> 1/1 (Original Resolution)
            LearnableUpsampleBlock(64, 32),

            # Final Projection to Depth (1 channel)
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Softplus() # Force positive depth
        )

        if mlp_weights_path is not None and os.path.exists(mlp_weights_path):
            state_dict = torch.load(mlp_weights_path, map_location='cpu')
            if 'model' in state_dict: 
                state_dict = state_dict['model']

            # Load everything (Backbone + Decoder)
            missing, unexpected = self.load_state_dict(state_dict, strict=False)

            if len(missing) == 0:
                logging.info("Success: Full model (Backbone + MLP) loaded.")
            else:
                logging.warning(f"Partial load. Missing keys: {len(missing)}")

    def forward(self, x):
        # 1. Extract Features (Frozen)
        # Input: [B, 3, H, W] -> Output: [B, 768, H/32, W/32]
        features = self.backbone(x)

        # 2. Decode & Upsample (Learnable)
        # Input: [B, 768, H/32, W/32] -> Output: [B, 1, H, W]
        depth_map = self.decoder(features)

        return depth_map
