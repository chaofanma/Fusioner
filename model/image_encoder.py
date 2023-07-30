from typing import List

import clip
import einops
import torch
import torch.nn as nn
from clip.model import LayerNorm, Transformer


class ClipViT(nn.Module):
    def __init__(self, tag: str, input_resolution: int, patch_size: int, width:int=768, layers:int=12, heads:int=12, output_dim:int=512, trainable=False):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        if isinstance(input_resolution, List):
            w, h = input_resolution
            num_patches = (w//patch_size)*(h//patch_size)
        else:
            num_patches = (input_resolution//patch_size)**2
        self.positional_embedding = nn.Parameter(scale * torch.randn(num_patches + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        
        self.trainable = trainable
    
    @property
    def dtype(self):
        return self.conv1.weight.dtype
    
    def forward_core(self, x: torch.Tensor):
        x = self.conv1(x.type(self.dtype))
        b, c, w, h = x.shape
        x = einops.rearrange(x, 'b c w h -> b (w h) c')
        cls = einops.repeat(self.class_embedding.to(x.dtype), 'c -> b n c', b=b, n=1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        return x
    
    def encode_img(self, x: torch.Tensor):
        x = self.forward_core(x)
        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj
        
        if not self.trainable:
            x = x.float()
        return x
    
    def forward(self, x: torch.Tensor):
        
        x = self.forward_core(x)
        x = self.ln_post(x[:, 1:, :])

        if self.proj is not None:
            x = x @ self.proj
        
        if not self.trainable:
            x = x.float()
        return x

