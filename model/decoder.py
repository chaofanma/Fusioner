from functools import partial
from typing import Tuple

import einops
import torch.nn as nn
from torch.nn import functional as F

from configs import args


class Decoder(nn.Module):
    def __init__(self, in_channels, wo):
        super(Decoder, self).__init__()
        
        self.conv_0 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv_4 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)

        self.norm_0 = nn.InstanceNorm2d(in_channels)
        self.norm_1 = nn.InstanceNorm2d(in_channels)
        self.norm_2 = nn.InstanceNorm2d(in_channels)
        self.norm_3 = nn.InstanceNorm2d(in_channels)
        self.output_dim = in_channels
        self.wo = wo


    def forward(self, inputs: Tuple):
        
        x, text = inputs
        
        x = einops.rearrange(x, 'b (wo ho) c -> b c wo ho', wo=self.wo)
            
        w, h = x.shape[-2:]
        
        x = self.conv_0(x)
        x = self.norm_0(x)
        x = F.relu(x, inplace=True)
        
        x = F.interpolate(x, size=(w*2, h*2), mode='bilinear')
        
        x = self.conv_1(x)
        x = self.norm_1(x)
        x = F.relu(x, inplace=True)
        
        x = F.interpolate(x, size=(w*4, h*4), mode='bilinear')
        
        x = self.conv_2(x)
        x = self.norm_2(x)
        x = F.relu(x, inplace=True)
        
        x = F.interpolate(x, size=(w*8, h*8), mode='bilinear')
        
        x = self.conv_3(x)
        x = self.norm_3(x)
        x = F.relu(x, inplace=True)
        
        x = self.conv_4(x)
        x = F.interpolate(x, size=(args.target_size[0], args.target_size[0]), mode='bilinear')
        
        return x, text
