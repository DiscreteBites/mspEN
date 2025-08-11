import numpy as np
import torch

from .autoencoder import NeurogramVAE
from .dataset import make_loader  

def trainAE( neurogram_path: str, phoneme_path: str ):    
    C = 150
    
    model = NeurogramVAE(n_bands=C, n_features=64, depth=4, latent_dim=256, msp_label_size=40)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # data
    neurogram = np.load( neurogram_path )   # shape [N_time, F]
    phonemes = np.load( phoneme_path )    # shape [N_time], ints in {-1..39}

    # build loader
    _, loader, _ = make_loader(neurogram, phonemes, batch_size=8, T=512, hop=512, min_labeled_ratio=0.6)

    model.train()
    for step, (x, y) in enumerate(loader):

        # x: (B, F, T) ; y: (B, 40)
        opt.zero_grad()
        recon, z, mu, logvar = model(x)

        # total loss with MSP
        L_rec = model.loss_recon(recon, x)
        L_kl = model.loss_kl(mu, logvar)
        weight = x.numel() / (y.numel() + z.numel())
        L_msp = model.loss_msp(y, z) * weight
        L = L_rec + L_kl + L_msp
        L.backward()
        opt.step()

        if step % 50 == 0:
            print(f"step {step}  total {L.item():.2f}  rec {L_rec.item():.2f}  kl {L_kl.item():.2f}  msp {L_msp.item():.2f}")