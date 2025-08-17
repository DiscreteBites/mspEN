
import torch
from torch import nn as nn

class AdaptiveNorm2d(nn.Module):
    """
    Adaptive Instance Normalization that learns different normalization 
    for spatial vs temporal dimensions
    """
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        
        # Separate learnable parameters for spatial and temporal normalization
        self.spatial_weight = nn.Parameter(torch.ones(num_features))
        self.spatial_bias = nn.Parameter(torch.zeros(num_features))
        self.temporal_weight = nn.Parameter(torch.ones(num_features))
        self.temporal_bias = nn.Parameter(torch.zeros(num_features))
        
        # Mixing parameter to combine spatial and temporal normalization
        self.mix_weight = nn.Parameter(torch.ones(num_features))
        
        # Fallback normalization for edge cases
        self.fallback_norm = nn.BatchNorm2d(num_features)
    
    def forward(self, x):
        # x shape: (batch, channels, height=150, width=50)
        batch_size, channels, height, width = x.shape
        
        # Use fallback for very small dimensions
        if height <= 1 and width <= 1:
            return self.fallback_norm(x)
        
        # ====================================================
        # Compute statistics along spatial dimension (height)
        # ====================================================

        if height > 1:
            spatial_mean = x.mean(dim=2, keepdim=True)  # Mean along neuron bundles
            spatial_var = x.var(dim=2, keepdim=True)    # Var along neuron bundles
            spatial_normalized = (x - spatial_mean) / torch.sqrt(spatial_var + self.eps)

            spatial_out = spatial_normalized * self.spatial_weight.view(1, -1, 1, 1) + \
                            self.spatial_bias.view(1, -1, 1, 1)
        else:
            spatial_out = x
        
        # ====================================================
        # Compute statistics along temporal dimension (width)
        # ====================================================

        if width > 1:
            temporal_mean = x.mean(dim=3, keepdim=True)  # Mean along time steps
            temporal_var = x.var(dim=3, keepdim=True)    # Var along time steps
            temporal_normalized = (x - temporal_mean) / torch.sqrt(temporal_var + self.eps)
            temporal_out = temporal_normalized * self.temporal_weight.view(1, -1, 1, 1) + \
                            self.temporal_bias.view(1, -1, 1, 1)
        else:
            temporal_out = x
        
        # ======================================================
        # Adaptive mixing of spatial and temporal normalization
        # ======================================================
        
        if height > 1 and width > 1:
            # Both dimensions available - use adaptive mixing
            mix_weight = torch.sigmoid(self.mix_weight.view(1, -1, 1, 1))
            return mix_weight * spatial_out + (1 - mix_weight) * temporal_out
        elif height > 1:
            # Only spatial dimension available
            return spatial_out
        elif width > 1:
            # Only temporal dimension available
            return temporal_out
        else:
            return self.fallback_norm(x)