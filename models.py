
import numpy as np 
import torch
from torch import nn

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, cond_embed_dim):
        
        super().__init__()

        # Convolution: in_channels -> out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Normalisation: groupnorm better for small batch sizes #TODO: check 8
        self.norm = nn.GroupNorm(8, out_channels)
        
        # Activation
        self.act = nn.SiLU() # or nn.ReLU() -> test, but silu might work better for diff. 

        # Project conditioning to out_channels
        self.cond_proj = nn.Linear(cond_embed_dim, out_channels)

    
    def forward(self, x, cond):
        # x: (batch, in_channels, H, W)
        # cond: (batch, cond_embed_dim)

        # apply conv and norm
        x = self.conv(x)
        x = self.norm(x)

        cond = self.cond_proj(cond)
        cond = cond[:, :, None, None]
        x = x + cond

        # activation
        x = self.act(x)

        return x 



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

        output = torch.cat([sines, cosines], dim=-1)

        return output 
    

class ConditioningMLP(nn.Module):

    def __init__(self, cond_embed_dim=256):
        super().__init__()

        self.cond_embed_dim = cond_embed_dim

        self.sin_embed = SinusoidalEmbedding(dim=self.cond_embed_dim)

        self.l1 = nn.Linear(512, 256)
        self.l2 = nn.ReLU()
        self.l3 = nn.Linear(256, 256)



    def forward(self, d_level, delta_t):
        """
        d_level: (batch,)
        delta_t: (batch,)
        return: (batch, embecond_embed_dimd_dim)
        """

        d_level_embed = self.sin_embed(d_level)
        delta_t_embed = self.sin_embed(delta_t)

        x = torch.cat([d_level_embed, delta_t_embed], dim=-1)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        return x




class UNet(nn.Module):
    def __init__(self, image_channels=1, cond_embed_dim=256):
        # TODO: inimage_channels_channels = 1 because we are working in grayscale. 


        super().__init__()

        in_channels = image_channels * 2
        out_channels = image_channels

        self.cond_mlp = ConditioningMLP(cond_embed_dim=cond_embed_dim)

        # encoder 
        self.enc1 = ConvBlock(in_channels, 64, cond_embed_dim)
        self.down1 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        self.enc2 = ConvBlock(64, 128, cond_embed_dim)
        self.down2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

        self.enc3 = ConvBlock(128, 256, cond_embed_dim)
        self.down3 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        # bottleneck
        self.bottleneck = ConvBlock(256, 512, cond_embed_dim)

        # decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec1 = ConvBlock(512 + 256, 256, cond_embed_dim)


        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec2 = ConvBlock(256 + 128, 128, cond_embed_dim)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec3 = ConvBlock(128 + 64, 64, cond_embed_dim)
        
        # Output
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)


    def forward(self, noisy_curr, prev_frame, d_level, delta_t):

        # get conditioning vector
        cond = self.cond_mlp(d_level, delta_t)

        # concatenate images
        x = torch.cat([noisy_curr, prev_frame], dim=1)

        e1 = self.enc1(x, cond)
        x = self.down1(e1)

        e2 = self.enc2(x, cond)
        x = self.down2(e2)

        e3 = self.enc3(x, cond)
        x = self.down3(e3)

        x = self.bottleneck(x, cond)

        # decoder (concaatenate skip connections)
        x = self.up1(x)
        x = torch.cat([x, e3], dim=1)
        x = self.dec1(x, cond)

        x = self.up2(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x, cond)

        x = self.up3(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec3(x, cond)

        x = self.out_conv(x)

        return x





model = UNet(image_channels=1, cond_embed_dim=256)
noisy = torch.randn(2, 1, 128, 128)
prev = torch.randn(2, 1, 128, 128)
d = torch.tensor([500.0, 100.0])
t = torch.tensor([3.0, 7.0])

out = model(noisy, prev, d, t)
print(out.shape)