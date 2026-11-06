import torch
import torch.nn as nn
from ultralytics.nn.modules import Detect

class DetectAux(Detect):
    """辅助检测头，仅在训练时使用"""
    def forward(self, x):
        if self.training:
            return super().forward([x])
        else:
            return []  # 推理时不输出