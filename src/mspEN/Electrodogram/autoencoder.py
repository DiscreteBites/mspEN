import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# ----------------------------
# Electrodogram (T x 2) → LSTM
# ----------------------------

class ElectroEncoder(nn.Module):
	"""
	Input:  x of shape (B, T, 2)
	Output: z of shape (B, latent_dim)
	"""
	def __init__(self, latent_dim: int = 8, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.0):
		super().__init__()
		self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_size, num_layers=num_layers,
		                    batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
		self.to_latent = nn.Linear(hidden_size, latent_dim)

	def forward(self, x):
		# x: (B, T, 2)
		_, (h_n, _) = self.lstm(x)        # h_n: (num_layers, B, H)
		h_last = h_n[-1]                  # (B, H)
		z = self.to_latent(h_last)        # (B, latent_dim)
		return z


class ElectroDecoder(nn.Module):
	"""
	Input:  z of shape (B, latent_dim), target length T
	Output: y of shape (B, T, 2)
	"""
	def __init__(self, latent_dim: int = 8, hidden_size: int = 32, num_layers: int = 1):
		super().__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers

		# Map latent to initial hidden & cell states
		self.init_h = nn.Linear(latent_dim, num_layers * hidden_size)
		self.init_c = nn.Linear(latent_dim, num_layers * hidden_size)

		# We condition via initial state; inputs are zeros
		self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
		self.readout = nn.Linear(hidden_size, 2)

	def forward(self, z, T: int):
		B = z.size(0)

		# Initial states from latent
		h0 = self.init_h(z).view(self.num_layers, B, self.hidden_size).contiguous()
		c0 = self.init_c(z).view(self.num_layers, B, self.hidden_size).contiguous()

		# Zero inputs; the model uses initial state to generate the sequence
		x0 = torch.zeros(B, T, 2, device=z.device, dtype=z.dtype)

		out, _ = self.lstm(x0, (h0, c0))     # (B, T, H)
		y = self.readout(out)                # (B, T, 2)
		return y