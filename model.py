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
        self.backbone = timm.create_model(arch, pretrained=False, features_only=True)

        # Load Safetensors and freeze backbone
        logger.info(f"Loading weights from {dino_weights_path}...")
        raw_weights = load_file(dino_weights_path)
        self.backbone.load_state_dict(raw_weights, strict=True)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # --- C. Learnable MLP Decoder ---
        enc_channels = self.backbone.feature_info.channels()

        # Stage 1: 1/32 -> 1/16
        # Input: feat32 (enc_channels[3])
        self.up1 = LearnableUpsampleBlock(enc_channels[3], enc_channels[2])
        
        # Stage 2: 1/16 -> 1/8
        # Input: up1 output + feat16 skip (enc_channels[2] + enc_channels[2])
        self.up2 = LearnableUpsampleBlock(enc_channels[2] * 2, enc_channels[1])
        
        # Stage 3: 1/8 -> 1/4
        # Input: up2 output + feat8 skip (enc_channels[1] + enc_channels[1])
        self.up3 = LearnableUpsampleBlock(enc_channels[1] * 2, enc_channels[0])
        
        # Stage 4: 1/4 -> 1/2
        # Input: up3 output + feat4 skip (enc_channels[0] + enc_channels[0])
        decoder_ch = enc_channels[0] // 2
        self.up4 = LearnableUpsampleBlock(enc_channels[0] * 2, decoder_ch)
        
        # Stage 5: 1/2 -> 1/1 (Original Resolution)
        # Note: ConvNeXt doesn't natively output a 1/2 stride feature map due to its 4x4 stem patchify.
        # So we just upsample without a skip connection here.
        self.up5 = LearnableUpsampleBlock(decoder_ch, 32)

        # Final Projection to Depth (1 channel)
        self.head = nn.Sequential(
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
        # Returns a list: [feat4, feat8, feat16, feat32]
        # Strides:         1/4    1/8    1/16    1/32
        features = self.backbone(x)
        feat4, feat8, feat16, feat32 = features

        # 2. Decode & Upsample with Skip Connections
        # 1/32 -> 1/16
        x = self.up1(feat32)
        x = torch.cat([x, feat16], dim=1)  # Concat skip connection
        
        # 1/16 -> 1/8
        x = self.up2(x)
        x = torch.cat([x, feat8], dim=1)   # Concat skip connection
        
        # 1/8 -> 1/4
        x = self.up3(x)
        x = torch.cat([x, feat4], dim=1)   # Concat skip connection
        
        # 1/4 -> 1/2
        x = self.up4(x)
        
        # 1/2 -> 1/1
        x = self.up5(x)

        # Final prediction
        depth_map = self.head(x)

        return depth_map
