import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# --------------------------------
# Neurogram (T x 150) → 1D CNN AE
# --------------------------------

class NeuroEncoder(nn.Module):
	"""
	Input:  x of shape (B, T, 150)
	Output: z of shape (B, latent_dim)
	Note: Conv1d expects (B, C, L) → we treat channels=150, length=T
	"""
	def __init__(self, latent_dim: int = 32):
		super().__init__()
		self.conv1 = nn.Conv1d(150, 64, kernel_size=3, padding=1)   # keep length
		self.bn1   = nn.BatchNorm1d(64)
		self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
		self.bn2   = nn.BatchNorm1d(32)
		# Fix the temporal dimension to a small constant with adaptive pooling (works for any T)
		self.pool  = nn.AdaptiveAvgPool1d(output_size=16)
		self.proj  = nn.Linear(32 * 16, latent_dim)

	def forward(self, x):
		# x: (B, T, 150) → (B, 150, T)
		x = x.transpose(1, 2)
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = self.pool(x)                    # (B, 32, 16)
		x = x.flatten(1)                    # (B, 32*16)
		z = self.proj(x)                    # (B, latent_dim)
		return z


class NeuroDecoder(nn.Module):
	"""
	Input:  z of shape (B, latent_dim), target length T
	Output: y of shape (B, T, 150)
	Strategy: project to (B, 32, 16), upsample to T, then refine & read out 150 channels
	"""
	def __init__(self, latent_dim: int = 32):
		super().__init__()
		self.expand = nn.Linear(latent_dim, 32 * 16)
		self.conv_up1 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
		self.bn_up1   = nn.BatchNorm1d(64)
		self.conv_out = nn.Conv1d(64, 150, kernel_size=3, padding=1)

	def forward(self, z, T: int):
		B = z.size(0)
		x = self.expand(z).view(B, 32, 16)     # (B, 32, 16)
		# Upsample temporal dim to T
		x = F.interpolate(x, size=T, mode='linear', align_corners=False)  # (B, 32, T)
		x = F.relu(self.bn_up1(self.conv_up1(x)))                         # (B, 64, T)
		x = self.conv_out(x)                                              # (B, 150, T)
		y = x.transpose(1, 2)                                             # (B, T, 150)
		return y