import torch
import torch.nn as nn
from .ssd_neck import SSDNeck

class SSDNeckWrapper(nn.Module):
    def __init__(self, in_channels=[512, 512, 1024]):
        super().__init__()
        self.ssd_neck = SSDNeck(in_channels=in_channels)

    def forward(self, x):
        outputs = self.ssd_neck(x)
        # 确保是 list 类型（YOLO parse_model 支持）
        if isinstance(outputs, tuple):
            outputs = list(outputs)
        return outputs
