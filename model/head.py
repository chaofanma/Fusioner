from typing import Tuple

import torch.nn as nn
from einops import parse_shape, rearrange


class Head(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, x:Tuple):
        features, cls_seg_feat = x
        patches = rearrange(features, "b c h w -> b (h w) c")

        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        masks = patches @ cls_seg_feat.transpose(1, 2)
        logits = rearrange(masks, "b (h w) n -> b n h w", **parse_shape(features, 'b _ h w'))
        
        logits = logits / self.temperature

        return logits
