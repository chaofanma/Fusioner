from typing import Tuple

import torch
import torch.nn as nn

from utils.model_utils import Block


class Fusion(nn.Module):
    def __init__(
        self,
        n_visual_token,
        n_cls,
        visual_dim,
        text_dim,
        n_layers,
        n_heads,
        d_model,
        d_ff,
        drop_path_rate,
        dropout,
    ):
        super().__init__()
        self.n_visual_token = n_visual_token
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )
        
        self.cls_emb = nn.Embedding(n_cls, d_model).weight
        
        self.visual_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, d_model),
            nn.LayerNorm(d_model),
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, d_model),
        )
        
        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        
        self.norm = nn.LayerNorm(d_model)


    def forward(self, inputs: Tuple):
        visual, text = inputs
        
        n_cls = text.shape[1]
        x = self.visual_proj(visual)
        text_feat = self.text_proj(text)
        
        text_feat = text_feat + self.cls_emb
        x = torch.cat((x, text_feat), 1)
        
        for blk in self.blocks:
            x, att = blk(x, return_attention=True)
        x = self.norm(x)

        patches, cls_seg_feat = x[:, : -(n_cls)], x[:, -n_cls :]
        
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes
        
        return patches, cls_seg_feat

