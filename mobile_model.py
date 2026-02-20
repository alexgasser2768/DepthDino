# add to model.py
import torch.nn as nn
import torch.nn.functional as F

class DWConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.act(self.dw_bn(self.dw(x)))
        x = self.act(self.pw_bn(self.pw(x)))
        return x


class UpFuse(nn.Module):
    """Upsample to skip resolution, concat skip, then DWConvBlock."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.block = DWConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class MobileNetV4DepthStudent(nn.Module):
    def __init__(
        self,
        backbone_name: str = "mobilenetv4_conv_small",
        pretrained: bool = True,
        out_indices=(1, 2, 3, 4),  # should correspond to strides 4,8,16,32
        decoder_ch: int = 96,
        return_feats: bool = False,
    ):
        super().__init__()
        self.return_feats = return_feats

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )

        ch = self.backbone.feature_info.channels()  # [c4, c8, c16, c32]
        c4, c8, c16, c32 = ch

        self.proj32 = DWConvBlock(c32, decoder_ch)
        self.up16 = UpFuse(decoder_ch, c16, decoder_ch)
        self.up8  = UpFuse(decoder_ch, c8, decoder_ch // 2)
        self.up4  = UpFuse(decoder_ch // 2, c4, decoder_ch // 4)

        self.refine = DWConvBlock(decoder_ch // 4, decoder_ch // 4)
        self.head = nn.Conv2d(decoder_ch // 4, 1, kernel_size=3, padding=1)

    def forward(self, x):
        feat4, feat8, feat16, feat32 = self.backbone(x)

        y = self.proj32(feat32)
        y = self.up16(y, feat16)
        y = self.up8(y, feat8)
        y = self.up4(y, feat4)

        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        y = self.refine(y)

        depth = F.relu(self.head(y))  # positive for log-based losses

        if self.return_feats:
            return depth, (feat4, feat8, feat16, feat32)
        return depth
