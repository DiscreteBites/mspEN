import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t

from .normalisation import AdaptiveNorm2d

class VGGBlock(nn.Module):
    """
    VGG-style block for encoder with joint spatiotemporal processing
    
    Handles conv → norm → activation → optional pooling
    """
    def __init__(self, 
        in_channels: int, out_channels: int, kernel_size: _size_2_t, 
        stride: _size_2_t, padding: _size_2_t | str =1, pool_size = None,
        norm_type: str | None = 'batch'
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

        # Normalization layer
        if norm_type == 'adaptive':
            self.norm = AdaptiveNorm2d(out_channels)
        elif norm_type == 'batch':
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = nn.Identity()

        self.activation = nn.LeakyReLU(0.2, inplace=True)

        self.pool = nn.MaxPool2d(pool_size) if pool_size else None
    
    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        
        if self.pool:
            x = self.pool(x)
            
        return x


class AdaptiveResize(nn.Module):
    """
    Adaptive interpolation layer
    """
    def __init__(self, target_size, mode='bilinear', align_corners=False):
        super().__init__()
        self.target_size = target_size
        self.mode = mode
        self.align_corners = align_corners
    
    def forward(self, x):
        return F.interpolate(x, size=self.target_size, 
                           mode=self.mode, align_corners=self.align_corners)
    
    def update_target_size(self, new_size):
        """Update target size for variable time dimension experiments"""
        self.target_size = new_size


class InvVGGBlock(nn.Module):
    """
    Inverse VGG-style block for decoder with joint spatiotemporal processing
    
    Handles upsample → conv → norm → activation OR transpose_conv → norm → activation
    """
    def __init__(self, 
        in_channels: int, out_channels: int, kernel_size: _size_2_t =3, padding: _size_2_t | str =1,
        upsample_factor=None, transpose_conv_params=None,
        norm_type: str | None = 'batch', final_layer=False
    ):
        super().__init__()
        
        self.upsample_factor = upsample_factor
        self.final_layer = final_layer
        
        if upsample_factor:
            # Upsample + Conv approach (more control)
            self.upsample = nn.Upsample(scale_factor=upsample_factor, mode='nearest')
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        elif transpose_conv_params:
            # Transposed convolution approach
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, **transpose_conv_params)
        else:
            # Regular convolution (for refinement layers)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        
        # Normalization (skip for final layer)
        if not final_layer:
            if norm_type == 'adaptive':
                self.norm = AdaptiveNorm2d(out_channels)
            elif norm_type == 'batch':
                self.norm = nn.BatchNorm2d(out_channels)
            else:
                self.norm = nn.Identity()
        else:
            self.norm = None
        
        # Activation
        if final_layer:
            self.activation = nn.Tanh()  # Bound final output
        else:
            self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        if self.upsample_factor:
            x = self.upsample(x)
    
        x = self.conv(x)
        
        if self.norm:
            x = self.norm(x)
        
        x = self.activation(x)
        return x
