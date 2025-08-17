from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mspEN.modules import VGGBlock, AdaptiveResize, InvVGGBlock, VAEType

class NeurogramEncoder(nn.Module):
    '''
    Encoder that progressively reduces neurograms to 1024-dimensional latent space
    
    Architecture progression:
    Input:  (batch, 1, 150, T)      # T ∈ {25, 50, 100, 200}
    Stage1: (batch, 64, 150, T)     # Early spatiotemporal features [BatchNorm]
    Stage2: (batch, 128, 75, T/2)   # Joint processing [BatchNorm] 
    Stage3: (batch, 256, 25, T/4)   # Asymmetric spatial emphasis [Adaptive]
    Stage4: (batch, 512, 5, T/8)    # High-level abstraction [Adaptive]
    Stage5: (batch, 1024, 1, 1)     # Global compression [Adaptive]
    '''
    
    def __init__(self, latent_dim: int = 1024, use_adaptive_norm: bool = True):
        super().__init__()

        self.latent_dim = latent_dim
        self.use_adaptive_norm = use_adaptive_norm

        def get_norm_type(stage):
            if not use_adaptive_norm:
                return 'batch'
            return 'batch' if stage <= 2 else 'adaptive'
        
        self.main = nn.Sequential(
            # Stage 1: Early spatiotemporal feature extraction
            # 150×T → 150×T (preserve resolution, learn basic patterns)
            VGGBlock(1, 64, kernel_size=(5, 3), padding=(2, 1), 
                norm_type=get_norm_type(1), stride=1),

            # Stage 2: Joint processing with first reduction  
            # 150×T → 75×T/2 (moderate spatial reduction, begin temporal compression)
            VGGBlock(64, 128, kernel_size=(3, 3), padding=1,
                norm_type=get_norm_type(2), stride=1, pool_size=(2, 2)),

            # Stage 3: Asymmetric reduction with spatial emphasis
            # 75×T/2 → 25×T/4 (aggressive spatial reduction)
            VGGBlock(128, 256, kernel_size=(7, 3), padding=(3, 1),
                    norm_type=get_norm_type(3), stride=(3, 2)),
            # Stage 4: High-level abstraction
            # 25×T/4 → 5×T/8 (continue aggressive spatial reduction)
            VGGBlock(256, 512, kernel_size=(5, 3), padding=(2, 1),
                    norm_type=get_norm_type(4), stride=(5, 2)),
            
            # Stage 5: Global compression 
            # 5×T/8 → 1×1 (force global representation)
            VGGBlock(512, 1024, kernel_size=(5, 3), padding=(2, 1),
                    norm_type=get_norm_type(5), stride=(5, 2)),
            
            # Handle any remaining spatial/temporal dims
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 150, 50) → neurogram input
        features = self.main(x)  # → (batch, final_channels, 1, 1)
        return features.squeeze(-1).squeeze(-1)

    def layer_summary(self, X_shape: Tuple[int, ...]):
        X = torch.randn(*X_shape)
        print(f'Input tensor: {X.shape}')
        for layer in self.main:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)


class NeurogramDecoder(nn.Module):
    """
    Decoder that reconstructs neurograms from 1024-dimensional latent space
    
    Architecture progression (reverse of encoder):
    Start:  (batch, 1024, 1, 1)     # Reshape for 2D processing
    Stage1: (batch, 512, 5, T/8)    # Initial spatial expansion [Adaptive]
    Stage2: (batch, 256, 25, T/4)   # Regional reconstruction [Adaptive]
    Stage3: (batch, 128, 75, T/2)   # Asymmetric expansion [Adaptive→Batch transition]
    Stage4: (batch, 64, 150, T)     # Local feature reconstruction [BatchNorm]
    Output: (batch, 1, 150, T)      # Final detail restoration [BatchNorm]
    """
   
    def __init__(self, latent_dim: int = 1024, target_time_dim: int = 50, use_adaptive_norm=True):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.target_time_dim = target_time_dim
        
        def get_norm_type(stage):
            if not use_adaptive_norm:
                return 'batch'
            return 'adaptive' if stage <= 3 else 'batch'
        
        # Create the adaptive resize layer
        self.adaptive_resize = AdaptiveResize(target_size=(150, target_time_dim))

        # Sequential decoder with tensor size annotations
        self.main = nn.Sequential(
            # Stage 1: Initial spatial expansion
            # 1×1 → 5×4 (begin spatial reconstruction)
            InvVGGBlock(1024, 512, 
                transpose_conv_params={'kernel_size': (5, 4), 'stride': 1, 'padding': 0},
                norm_type=get_norm_type(1)
            ),
            
            # Stage 2: Regional reconstruction 
            # 5×4 → 25×T/4 (continue spatial expansion, handle temporal adaptively)
            InvVGGBlock(512, 256,
                upsample_factor=(5, 2),  # Approximate upsampling
                norm_type=get_norm_type(2)
            ),
            
            # Stage 3: Asymmetric expansion
            # 25×T/4 → 75×T/2 (reverse encoder's asymmetric reduction)
            InvVGGBlock(256, 128,
                upsample_factor=(3, 2),
                norm_type=get_norm_type(3)
            ),
            
            # Stage 4: Local feature reconstruction
            # 75×T/2 → 150×T (restore full spatial resolution)
            InvVGGBlock(128, 64,
                upsample_factor=(2, 2),
                norm_type=get_norm_type(4)
            ),

            # Adaptive resize to target temporal dimension
            self.adaptive_resize,

            # Final layer: Detail restoration
            # 150×T → 150×T (refine details, final output)
            InvVGGBlock(64, 1,
                kernel_size=(5, 3), padding=(2, 1),
                norm_type=get_norm_type(5),
                final_layer=True
            )
        )

    def forward(self, z: torch.Tensor):
        z = z.view(z.size(0), self.latent_dim, 1, 1)  # → (batch, initial_channels, 1, 1)
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
    def __init__(self, latent_dim: int = 1024):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = NeurogramEncoder(latent_dim = self.latent_dim)
        self.decoder = NeurogramDecoder(latent_dim = self.latent_dim)

        # Intermediate mu and logvar latent parameters
        self.fc1 = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc2 = nn.Linear(self.latent_dim, self.latent_dim)
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc1(h)
        logvar = self.fc2(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
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
        
        Loss = l_recon + l_vae
        return Loss, l_recon.item(), l_vae.item()

    # def loss(self, 
    #     predict: torch.Tensor, target: torch.Tensor, 
    #     z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor,
    #     label: torch.Tensor | None
    # ):
    #     L_rec = self._loss_recon(predict, target)
    #     L_vae = self._loss_vae(mu, logvar)

    #     stats = {
    #         'reconstruction loss': L_rec.item(),
    #         'VAE loss': L_vae.item()
    #     }
        
    #     if self.M is not None and label is not None:
    #         L_msp = self._loss_msp(label, z)
    #         _msp_weight = target.numel()/(label.numel()+z.numel())

    #         Loss = L_rec + L_vae + L_msp * _msp_weight
    #         stats['MSP loss'] = L_msp.item()
    #         stats['MSP weight'] = _msp_weight
    #     else:
    #         Loss = L_rec + L_vae
        
    #     stats['total loss'] = Loss.item()
    #     return Loss, stats

    # def acc(self, z, l):
    #     zl = z @ self.M.t()
    #     a = zl.clamp(-1, 1)*l*0.5+0.5
    #     return a.round().mean().item()

    # def predict(self, x, new_ls=None, weight=1.0):
    #     z, _ = self.encode(x)
    #     if new_ls is not None:
    #         zl = z @ self.M.t()
    #         d = torch.zeros_like(zl)
    #         for i, v in new_ls:
    #             d[:,i] = v*weight - zl[:,i]
    #         z += d @ self.M
    #     prod = self.decoder(z)
    #     return prod

    # def predict_ex(self, x, label, new_ls=None, weight=1.0):
    #     return self.predict(x,new_ls,weight)
