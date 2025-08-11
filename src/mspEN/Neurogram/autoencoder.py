import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeurogramEncoder(nn.Module):
    '''
    Encoder for neurograms shaped (B, C, T):
    - Bands/channels C are treated as input channels
    - Temporal Conv1d with stride=2 compresses time
    - AdaptiveAvgPool1d -> time=1 gives a single feature vector per example
    - Two linear heads output μ and log sigma^2  for the VAE latent
    '''
    
    def __init__(self, n_bands: int = 150, n_features: int = 64, depth: int = 4, latent_dim: int = 256):
        super().__init__()

        layers = []
        in_channels = n_bands
        channels = n_features

        for _ in range(depth):
            layers += [
                nn.Conv1d(in_channels, channels, kernel_size=5, stride=2, padding=2, bias=False),
                nn.BatchNorm1d(channels),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_channels = channels
            channels = min(channels * 2, 1024)  # cap capacity as time shrinks
        
        self.backbone = nn.Sequential(*layers)
        self.out_channels = in_channels  # channels at the end of the conv stack
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mu = nn.Linear(self.out_channels, latent_dim)
        self.logvar = nn.Linear(self.out_channels, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        # x: (B, F, T)
        h = self.backbone(x)              # (B, C, T')
        Tp = h.shape[-1]                  # compressed time after strides
        hg = self.pool(h).squeeze(-1)     # (B, C)
        mu = self.mu(hg)                  # (B, H)
        logvar = self.logvar(hg)          # (B, H)
        return h, mu, logvar, Tp


class NeurogramDecoder(nn.Module):
    """
    Decoder mirrors the encoder with ConvTranspose1d:
    - Latent z -> linear -> seed feature map (C_seed, 1)
    - Deconvolution stack (stride=2) to expand time
    - Final layer outputs n_bands with Sigmoid (assuming inputs in [0,1])
    """
    def __init__(self, n_bands: int, n_features: int = 64, depth: int = 4, latent_dim: int = 256, enc_out_ch: int = 512):
        super().__init__()

        self.depth = depth
        self.seed_ch = enc_out_ch
        self.latent_to_seed = nn.Linear(latent_dim, self.seed_ch)

        # Mirror channel flow (reverse)
        chs = [enc_out_ch]
        channels = enc_out_ch
        for _ in range(depth - 1):
            channels = max(channels // 2, n_features)
            chs.append(channels)
        chs.append(n_bands)  # final out channels

        ups = []
        for i in range(depth):
            in_channels = chs[i]
            # for the last block, we jump directly to n_bands
            out_channels = chs[i + 1] if i + 1 < len(chs) - 1 else chs[-1]
            last = (i == depth - 1)
            ups.append(
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
            )
            if not last:
                ups += [nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True)]
        self.ups = nn.Sequential(*ups)
        self.out_act = nn.Sigmoid()

    def forward(self, z: torch.Tensor, T_target: int, T_seed: int) -> torch.Tensor:
        # z: (B, H)
        seed = self.latent_to_seed(z).unsqueeze(-1)  # (B, C_seed, 1)

        # Light learned upsampling to reach ~T_seed before the main stack.
        # Use a simple nearest upsample repeatedly to the required length,
        # then a 1x1 conv to mix channels back (kept simple here).
        x = seed
        while x.size(-1) < T_seed:
            # double length (nearest) to approach T_seed
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            if x.size(-1) > T_seed:
                break
        # Trim if we overshoot
        if x.size(-1) > T_seed:
            x = x[..., :T_seed]

        # Main upsampling stack
        x = self.ups(x)

        # Crop/pad to exactly T_target
        if x.size(-1) > T_target:
            x = x[..., :T_target]
        elif x.size(-1) < T_target:
            pad = T_target - x.size(-1)
            x = F.pad(x, (0, pad))
        return self.out_act(x)


class NeurogramVAE(nn.Module):
    """
    Full VAE for neurograms (B, C, T).
    - depth: # of stride-2 downsampling/upsampling stages
    - n_features: base channels
    - latent_dim: size of latent vector
    - msp_label_size: if set, enables MSP-style latent-label alignment
    """
    def __init__(
        self,
        n_bands: int,
        n_features: int = 64,
        depth: int = 4,
        latent_dim: int = 256,
        msp_label_size: Optional[int] = None
    ):
        super().__init__()
        self.enc = NeurogramEncoder(n_bands, n_features=n_features, depth=depth, latent_dim=latent_dim)
        self.dec = NeurogramDecoder(n_bands, n_features=n_features, depth=depth, latent_dim=latent_dim, enc_out_ch=self.enc.out_channels)
        self.latent_dim = latent_dim
        self.n_bands = n_bands
        self.depth = depth

        self.M: Optional[torch.Tensor] = None
        if msp_label_size is not None:
            self.M = nn.Parameter(torch.empty(msp_label_size, latent_dim))
            nn.init.xavier_normal_(self.M)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        return self.enc(x)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Pad time to multiple of 2^depth so the stride-2 pyramid is clean
        B, C, T = x.shape
        factor = 2 ** self.depth
        T_pad = math.ceil(T / factor) * factor
        if T_pad != T:
            xp = F.pad(x, (0, T_pad - T))
        else:
            xp = x

        h, mu, logvar, Tp = self.encode(xp)
        z = self.reparameterize(mu, logvar)
        recon = self.dec(z, T_target=T, T_seed=Tp)  # crop back to original T
        return recon, z, mu, logvar

    @staticmethod
    def loss_recon(pred: torch.Tensor, target: torch.Tensor, reduction: str = 'sum') -> torch.Tensor:
        """
        reduction: 'sum' (VAE-friendly default) or 'mean'
        """
        return F.mse_loss(pred, target, reduction=reduction)

    @staticmethod
    def loss_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def loss_msp(
        self,
        labels: torch.Tensor,                 # (B, 40) soft multi-labels in [0,1]
        z: torch.Tensor,                      # (B, H)
        class_weights: Optional[torch.Tensor] = None,  # (40,) or None
        symmetric: bool = True
    ) -> torch.Tensor:
        """
        Matrix Subspace Projection loss with optional per-class weighting.
        - Label-space term: || z M^T - labels ||^2, weighted per class.
        - Latent-space term: || labels M - z ||^2 (unweighted by default).
        """
        assert self.M is not None, "MSP requested but M not initialized"
        pred_labels = z @ self.M.t()          # (B, 40)
        if class_weights is not None:
            # ensure shape (1, 40) for broadcast over batch
            w = class_weights.view(1, -1).to(z.device, z.dtype)
            L1 = ((pred_labels - labels) ** 2 * w).sum()
        else:
            L1 = F.mse_loss(pred_labels, labels, reduction="sum")
        if symmetric:
            L2 = F.mse_loss(labels @ self.M, z, reduction="sum")
        else:
            L2 = z.new_zeros(())
        return L1 + L2

    def total_loss(
        self,
        x: torch.Tensor,                      # (B, F, T)
        labels: Optional[torch.Tensor] = None,# (B, 40) or None
        beta: float = 1.0,
        lambda_msp: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,  # (40,) or None
        scale_msp: str = 'auto',              # 'auto' | 'none' | float
        recon_reduction: str = 'sum'          # 'sum' | 'mean'
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        recon, z, mu, logvar = self.forward(x)

        # core terms
        L_rec = self.loss_recon(recon, x, reduction=recon_reduction)
        L_kl = self.loss_kl(mu, logvar) * beta

        # MSP (optional)
        if (labels is not None) and (self.M is not None):
            L_msp_raw = self.loss_msp(labels, z, class_weights=class_weights, symmetric=True)
            # scale MSP to balance with reconstruction
            if scale_msp == 'auto':
                scale = x.numel() / (labels.numel() + z.numel())
            elif scale_msp == 'none':
                scale = 1.0
            else:
                # user-specified numeric scale
                scale = float(scale_msp)
            L_msp = lambda_msp * scale * L_msp_raw
        else:
            L_msp_raw = torch.tensor(0., device=x.device)
            L_msp = torch.tensor(0., device=x.device)

        L = L_rec + L_kl + L_msp
        stats = {
            "recon": float(L_rec.item()),
            "kl": float(L_kl.item()),
            "msp_raw": float(L_msp_raw.item()),
            "msp": float(L_msp.item()),
            "beta": float(beta),
            "lambda_msp": float(lambda_msp)
        }
        return L, stats
    
    def predict(self, x: torch.Tensor, new_ls: Optional[list] = None, weight: float = 1.0) -> torch.Tensor:
        """
        Optionally steer latent with target labels new_ls = [(idx, value), ...]
        Returns a reconstructed neurogram with the steered latent.
        """
        recon, z, mu, logvar = self.forward(x)
        if (new_ls is not None) and (self.M is not None):
            zl = z @ self.M.t()
            d = torch.zeros_like(zl)
            for i, v in new_ls:
                d[:, i] = v * weight - zl[:, i]
            z = z + d @ self.M
            # Re-decode with same T and T_seed estimate
            B, C, T = x.shape
            T_seed = max(1, T // (2 ** self.depth))
            recon = self.dec(z, T_target=T, T_seed=T_seed)
        return recon


# -----------------------
# Minimal smoke test
# -----------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    B, C, T = 4, 150, 512  # 500 Hz -> ~1.024 s at T=512
    model = NeurogramVAE(n_bands=C, n_features=64, depth=4, latent_dim=256, msp_label_size=None)
    
    x = torch.rand(B, C, T)  # normalized neurograms in [0,1]
    recon, z, mu, logvar = model(x)
    print("Input shape :", x.shape)
    print("Recon shape :", recon.shape)
    print("Latent z    :", z.shape)

    L, stats = model.total_loss(x)
    print("Loss (total):", float(L.item()))
    print("Loss parts  :", stats)

    # Example training step (sketch)
    # opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # for step in range(1000):
    # 	opt.zero_grad()
    # 	L, _ = model.total_loss(x, beta=1.0)
    # 	L.backward()
    # 	opt.step()
