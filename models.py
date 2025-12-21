
import numpy as np 
import torch
from torch import nn

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, time_embed_dim):
        pass


class SinusoidalEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim 

    
    def forward(self, x):
        
        # for scalar value x and embedding dimension dim, you create dim/2 pairs of (sin, cos)
        
        i = torch.arange(0, self.dim, 2, device =x.device, dtype=x.dtype)
        freqs = 10000 ** (i / self.dim) 

        args = x[:, None] / freqs[None, :]

        sines = torch.sin(args)
        cosines = torch.cos(args)

        output = torch.concat([sines, cosines], dim=-1)

        return output 
    

embed = SinusoidalEmbedding(dim=256)
x = torch.tensor([500.0, 100.0])
out = embed(x)

print(out.shape)