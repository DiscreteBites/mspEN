from typing import Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from mspEN.modules import VAEType

class NeurogramEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            # Layer 1: Decrease spatial dimensions
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: Further decrease dimensions
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        
            # Layer 3: Downsample
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: Downsample
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 5: More downsampling
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 6: Downsampling to target size
            nn.Conv2d(1024, 1024, kernel_size=(4,3), stride=(1,1), padding=0, bias=False),
        )

    def forward(self, input: torch.Tensor):
        return self.main(input).squeeze(3).squeeze(2)

    def layer_summary(self, X_shape: Tuple[int, ...]):
        X = torch.randn(*X_shape)
        print(f'Input tensor: {X.shape}')
        for layer in self.main:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)


class NeurogramDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            # Layer 1: Increase spatial dimensions
            nn.ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Layer 2: Further increase dimensions
            nn.ConvTranspose2d(512, 256, kernel_size=(5, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        
            # Layer 3: Upsample
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Layer 4: Upsample
            nn.ConvTranspose2d(128, 64, kernel_size=(4, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Layer 5: More Upsampling
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 3), stride=(2, 2), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Layer 6: Last Upsampling to Target Size
            nn.ConvTranspose2d(32, 1, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0)),
            nn.Sigmoid()
        )
    
    def forward(self, z: torch.Tensor):
        z = z.unsqueeze(2).unsqueeze(2)
        output = self.main(z)
        return output

    def layer_summary(self, X_shape):
        X = torch.randn(*X_shape)
        print(f'Input tensor: {X.shape}')
        for layer in self.main:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)
class NeurogramVAE(VAEType):
    """
    Full VAE for neurograms (B, C, T).
    """    
    def __init__(self, beta_kld: float = 1):
        super().__init__()
        
        # Latent dimension hard coded to autoencoder architecture
        self.latent_dim = 1024
        self.beta_kld = beta_kld

        self.encoder = NeurogramEncoder()
        self.decoder = NeurogramDecoder()

        # Intermediate mu and logvar latent parameters
        self.fc1 = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc2 = nn.Linear(self.latent_dim, self.latent_dim)

    def update_hyperparameters(self, epoch: int):
        return
    
    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc1(h)
        logvar = self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        prod = self.decoder(z)
        return prod
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, logvar = self.encode(x)
        prod = self.decoder(z)
        return prod, z, mu, logvar
    
    @staticmethod
    def _loss_vae(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # https://arxiv.org/abs/1312.6114
        # KLD = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD

    @staticmethod
    def _loss_recon(predict: torch.Tensor, orig: torch.Tensor):
        batch_size = predict.shape[0]
        a = predict.view(batch_size, -1)
        b = orig.view(batch_size, -1)
        L = F.mse_loss(a, b, reduction='sum')
        return L
    
    def loss(self, predict: torch.Tensor, orig: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
        l_recon = self._loss_recon(predict, orig)
        l_vae = self._loss_vae(mu, logvar)
        return l_recon, self.beta_kld *  l_vae