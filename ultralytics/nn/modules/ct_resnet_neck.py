import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv


class CTResBlock(nn.Module):
    """CTResNet 基础残差块"""
    def __init__(self, c1, c2=None, shortcut=True, *args, **kwargs):
        super().__init__()
        if c2 is None:
            c2 = c1
        self.conv1 = Conv(c1, c2, 3, 1)
        self.conv2 = Conv(c2, c2, 3, 1)
        self.shortcut = shortcut and (c1 == c2)
        if not self.shortcut:
            self.shortcut_conv = Conv(c1, c2, 1)

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return (self.shortcut_conv(x) if not self.shortcut else x) + y


class CTResFuse(nn.Module):
    """跨层特征融合模块"""
    def __init__(self, c1, c2):
        super().__init__()
        self.reduce = Conv(c1, c2, 1, 1)
        self.ctblock = CTResBlock(c2 * 2, c2)
        self.out = Conv(c2, c2, 3, 1)

    def forward(self, x_high, x_low):
        # 上采样高层特征到低层尺寸 (使用 bilinear 避免 off-by-one)
        x_high = F.interpolate(self.reduce(x_high), size=x_low.shape[-2:], mode='bilinear', align_corners=False)
        fused = torch.cat([x_low, x_high], dim=1)
        out = self.ctblock(fused)
        return self.out(out)


class CTResNetNeck(nn.Module):
    """
    CTResNetNeck：top-down + bottom-up 双向特征融合
    输入: [P3, P4, P5]
    输出: [P3_out, P4_out, P5_out] (对齐 YOLO Head 尺寸比例)
    """
    def __init__(self, channels=(256, 512, 1024), debug=False):
        super().__init__()
        c3, c4, c5 = channels
        self.debug = debug

        # Top-down 路径
        self.reduce_p5 = Conv(c5, c4, 1, 1)
        self.fuse_p4 = CTResFuse(c4, c4)  # P5->P4
        self.fuse_p3 = CTResFuse(c4, c3)  # P4->P3

        # Bottom-up 路径
        self.down_p3 = Conv(c3, c3, 3, 2)
        self.enhance_p4 = CTResBlock(c3 + c4, c4)

        self.down_p4 = Conv(c4, c4, 3, 2)
        self.enhance_p5 = CTResBlock(c4 + c5, c5)

    def forward(self, features):
        x3, x4, x5 = features  # 分别对应 P3, P4, P5

        # -------- Top-down 融合 --------
        p5_reduced = self.reduce_p5(x5)
        p4_td = self.fuse_p4(p5_reduced, x4)
        p3_td = self.fuse_p3(p4_td, x3)

        # -------- Bottom-up 融合 --------
        p3_down = self.down_p3(p3_td)
        p4_bu = self.enhance_p4(torch.cat([p4_td, p3_down], dim=1))

        p4_down = self.down_p4(p4_bu)
        p5_bu = self.enhance_p5(torch.cat([p5_reduced, p4_down], dim=1))

        # -------- 尺度校正 (确保与 YOLO 检测头匹配) --------
        # P3: 1/8, P4: 1/16, P5: 1/32
        p3_out = p3_td
        p4_out = F.interpolate(p4_bu, size=(p3_out.shape[-2] // 2, p3_out.shape[-1] // 2), mode='nearest')
        p5_out = F.interpolate(p5_bu, size=(p3_out.shape[-2] // 4, p3_out.shape[-1] // 4), mode='nearest')

        if self.debug:
            print(f"[CTResNeck] Output shapes: P3={p3_out.shape}, P4={p4_out.shape}, P5={p5_out.shape}")

        return [p3_out, p4_out, p5_out]
