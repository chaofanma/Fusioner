from typing import List, Union

import clip
import numpy as np
import torch
import torch.nn as nn
from clip.model import LayerNorm, Transformer

from configs import args


class ClipText(nn.Module):
    def __init__(self,
                tag:str,
                 embed_dim:int=512,
                 context_length:int=77,
                 vocab_size:int=49408,
                 transformer_width:int=512,
                 transformer_heads:int=8,
                 transformer_layers:int=12
                 ):
        super().__init__()

        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.embed_dim = embed_dim
        
        self.is_model_fp16 = True
       
        self.template_list = [
            'a bad photo of the {category}.',
            'a photo of the large {category}.',
            'a photo of the small {category}.',
            'a cropped photo of a {category}.',
            'This is a photo of a {category}',
            'This is a photo of a small {category}',
            'This is a photo of a medium {category}',
            'This is a photo of a large {category}',

            'This is a masked photo of a {category}',
            'This is a masked photo of a small {category}',
            'This is a masked photo of a medium {category}',
            'This is a masked photo of a large {category}',

            'This is a cropped photo of a {category}',
            'This is a cropped photo of a small {category}',
            'This is a cropped photo of a medium {category}',
            'This is a cropped photo of a large {category}',

            'A photo of a {category} in the scene',
            'a bad photo of the {category} in the scene',
            'a photo of the large {category} in the scene',
            'a photo of the small {category} in the scene',
            'a cropped photo of a {category} in the scene',
            'a photo of a masked {category} in the scene',
            'There is a {category} in the scene',
            'There is the {category} in the scene',
            'This is a {category} in the scene',
            'This is the {category} in the scene',
            'This is one {category} in the scene',

            'There is a masked {category} in the scene',
            'There is the masked {category} in the scene',
            'This is a masked {category} in the scene',
            'This is the masked {category} in the scene',
            'This is one masked {category} in the scene',
        ]
        
    @property
    def dtype(self):
        blk = self.transformer.resblocks[0]
        return blk.mlp[0].weight.dtype    
    
    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask
    
    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, text:List[str]) -> torch.Tensor:
        return_embeds = []
        for class_name in text:
            class_embed = self.get_embed(class_name)
            return_embeds.append(class_embed)
        return_embeds = torch.stack(return_embeds, dim=0)
        
        if self.is_model_fp16:
            return_embeds = return_embeds.float()
            
        return return_embeds
    
    def get_embed(self, class_name:str):
        texts = [template.format(category=class_name) for template in self.template_list]
        texts = self.tokenize(texts).to(args.device)
        all_text_embed = self.encode_text(texts)
        all_text_embed /= all_text_embed.norm(dim=-1, keepdim=True)
        avg_text_embed = all_text_embed.mean(dim=0)
        avg_text_embed /= avg_text_embed.norm()
        return avg_text_embed.float()
 
    def tokenize(self, text: Union[str, List[str]]):
        return clip.tokenize(text).to(args.device)

