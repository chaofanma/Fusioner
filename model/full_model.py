import math
from collections import OrderedDict

import clip
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torchvision.transforms.functional import hflip

from configs import args

from .decoder import Decoder
from .fusion import Fusion
from .head import Head
from .image_encoder import ClipViT
from .text_encoder import ClipText


class FullModel(nn.Module):
    def __init__(self, base_size=520, crop_size=480, flip=True, scales=[1.0]):
        super().__init__()
        
        self.nclass = 1
        self.base_size = base_size
        self.crop_size = crop_size
        self.scales = scales
        self.flip = flip
        self.pad_mean = [0.5, 0.5, 0.5]
        self.pad_std = [0.5, 0.5, 0.5]
        self.up_kwargs = {'mode': 'bilinear', 'align_corners': True}

        vit14_text_archi = {'tag': 'ViT-L/14', 'embed_dim': 768, 'context_length': 77, 'vocab_size': 49408, 'transformer_width': 768, 'transformer_heads': 12, 'transformer_layers': 12}
        self.text_model = ClipText(**vit14_text_archi)
        self.text_dim = vit14_text_archi['embed_dim']

        vit14_img_archi = {'tag': 'ViT-L/14', 'output_dim': 768, 'input_resolution': 224, 'layers': 24, 'width': 1024, 'heads': 16, 'patch_size': 14}
        self.patch_size = vit14_img_archi['patch_size']
        self.img_model = ClipViT(**vit14_img_archi)
        self.visual_dim = vit14_img_archi['output_dim']
        self.dtype = self.img_model.conv1.weight.dtype

        downsampled_w = args.target_size[0]//self.patch_size if not isinstance(args.target_size, (int, float)) else args.target_size//self.patch_size
        
        self.fusion = Fusion(
            n_visual_token=downsampled_w**2,
            n_cls=self.nclass,
            visual_dim=self.visual_dim,
            text_dim=self.text_dim,
            n_layers=args.transformer_layers,
            n_heads=args.transformer_heads,
            d_model=args.transformer_width,
            d_ff=4*args.transformer_width,
            drop_path_rate=args.drop_path_rate,
            dropout=args.transformer_dropout,
        )    
        self.decoder = Decoder(args.transformer_width, downsampled_w)
        self.head = Head(temperature=args.temperature)
        
        self.init_weight()
    
    
    def init_weight(self):
        model, _ = clip.load('ViT-L/14', device='cpu')
        state_dict = model.state_dict()
        img_state_dict = OrderedDict()
        text_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('visual'):
                k = ".".join(k.split('.')[1:])
                img_state_dict[k] = v
            else:
                text_state_dict[k] = v
        self.img_model.load_state_dict(img_state_dict)
        self.text_model.load_state_dict(text_state_dict)

    
    def forward(self, img, text, batched_text=False):
        
        x = self.encoder_forward(img, text, batched_text)
        x = self.fusion(x)
        x = self.decoder(x)
        x = self.head(x)
        
        return x
    
    
    def encoder_forward(self, img, text, batched_text=False):
        B, C, W, H = img.shape
        
        if batched_text:
            text_encode = []
            for text_this_batch in text:
                text_encode.append(self.text_model(text_this_batch))
            text_encode = torch.stack(text_encode, dim=0)
        else:
            text_encode = self.text_model(text)
            text_encode = einops.repeat(text_encode, 'k d -> b k d', b=B)
        img_encode = self.img_model(img.type(self.dtype))

        img_encode = img_encode / img_encode.norm(dim=-1, keepdim=True)

        return img_encode, text_encode
    
  
    @torch.no_grad()
    def multi_scale_eval(self, image, text):
        batch, _, h, w = image.size()
        assert(batch == 1)
        stride_rate = 2.0/3.0
        crop_size = self.crop_size
        stride = int(crop_size * stride_rate)
        scores = image.new().resize_(batch,self.nclass,h,w).zero_().to(image.device)

        for scale in self.scales:
            long_size = int(math.ceil(self.base_size * scale))
            if h > w:
                height = long_size
                width = int(1.0 * w * long_size / h + 0.5)
                short_size = width
            else:
                width = long_size
                height = int(1.0 * h * long_size / w + 0.5)
                short_size = height
            cur_img = image
            long_size, short_size = (h, w) if h > w else (w, h)
            height, width = h, w
            if long_size <= crop_size:
                pad_img = pad_image(cur_img, self.pad_mean, self.pad_std, crop_size)
                outputs = module_inference(self, pad_img, text, self.flip)
                outputs = crop_image(outputs, 0, height, 0, width)
            else:
                if short_size < crop_size:
                    pad_img = pad_image(cur_img, self.pad_mean, self.pad_std, crop_size)
                else:
                    pad_img = cur_img
                _,_,ph,pw = pad_img.size()
                assert(ph >= height and pw >= width)
                h_grids = int(math.ceil(1.0 * (ph-crop_size)/stride)) + 1
                w_grids = int(math.ceil(1.0 * (pw-crop_size)/stride)) + 1
                outputs = image.new().resize_(batch,self.nclass,ph,pw).zero_().to(image.device)
                count_norm = image.new().resize_(batch,1,ph,pw).zero_().to(image.device)
                for idh in range(h_grids):
                    for idw in range(w_grids):
                        h0 = idh * stride
                        w0 = idw * stride
                        h1 = min(h0 + crop_size, ph)
                        w1 = min(w0 + crop_size, pw)
                        crop_img = crop_image(pad_img, h0, h1, w0, w1)
                        pad_crop_img = pad_image(crop_img, self.pad_mean, self.pad_std, crop_size)
                        output = module_inference(self, pad_crop_img, text, self.flip)
                        outputs[:,:,h0:h1,w0:w1] += crop_image(output, 0, h1-h0, 0, w1-w0)
                        count_norm[:,:,h0:h1,w0:w1] += 1
                assert((count_norm==0).sum()==0)
                outputs = outputs / count_norm
                outputs = outputs[:,:,:height,:width]
            score = outputs
            scores += score

        scores /= len(self.scales)
        return scores


def module_inference(module, image, text, flip=True):
    if flip:
        in_ = torch.cat([image, flip_image(image)], dim=0)
        out_ = module(in_, text)
        output = (out_[0:1] + flip_image(out_[1:2]))/2
    else:
        output = module(image, text)
    return output

def resize_image(img, h, w, **up_kwargs):
    return F.interpolate(img, (h, w), **up_kwargs)

def pad_image(img, mean, std, crop_size):
    b,c,h,w = img.size()
    assert(c==3)
    padh = crop_size - h if h < crop_size else 0
    padw = crop_size - w if w < crop_size else 0
    pad_values = -np.array(mean) / np.array(std)
    img_pad = img.new().resize_(b,c,h+padh,w+padw)
    for i in range(c):
        img_pad[:,i,:,:] = F.pad(img[:,i,:,:], (0, padw, 0, padh), value=pad_values[i])
    assert(img_pad.size(2)>=crop_size and img_pad.size(3)>=crop_size)
    return img_pad

def crop_image(img, h0, h1, w0, w1):
    return img[:,:,h0:h1,w0:w1]

def flip_image(img):
    img_flip = hflip(img)
    return img_flip
