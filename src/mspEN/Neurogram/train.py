import numpy as np
import torch
from tqdm import tqdm

from .autoencoder import NeurogramVAE
from .dataset import make_loader

def trainAE(
    data_path: str, 
    phoneme_path: str, 
    file_identifier_out: str = "neuro_vae_only", 
    epochs: int = 100,
    batch_size: int = 20
):

    # --- setup
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training AE with device:", device)
    
    # --- data
    neurogram = np.load( data_path )   # [N_time, F]
    phonemes = np.load( phoneme_path )     # [N_time] in {0..39}
    T = 128
    ds, loader  = make_loader(
        neurogram, phonemes,
        batch_size=batch_size, T=T, hop=int(T/4),
        shuffle=True, num_workers=0, smooth_alpha=0.0
    )
    class_w = ds.class_weights(device)

    # --- model/optim
    model = NeurogramVAE(
        nc=150,
        nf=64,
        depth=6,
        latent_dim=1024,
        msp_label_size=None
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.GradScaler(device='cuda', enabled=torch.cuda.is_available())
    
    # --- train
    model.train()
    global_step = 0
    best_loss = float("inf")

    for epoch in range(epochs):
        epoch_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        for step, (x, y) in enumerate(epoch_bar):
            x = x.to(device, non_blocking=True)   # (B, F, T)
            y = y.to(device, non_blocking=True)   # (B, 40)
            
            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type='cuda', 
                dtype=torch.float16, 
                enabled=(device.type == 'cuda')
            ):
                L, stats = model.total_loss(
                    x, labels=y,
                    beta=1.0,
                    lambda_msp=1.0,
                    class_weights=class_w,
                    scale_msp='auto',
                    recon_reduction='sum'
                )
            
            scaler.scale(L).backward()
            scaler.step(opt)
            scaler.update()
            
            # update tqdm bar postfix
            epoch_bar.set_postfix({
                "total": f"{L.item():.2f}",
                "rec": f"{stats['recon']:.2f}",
                "kl": f"{stats['kl']:.2f}",
                "msp": f"{stats['msp']:.2f}"
            })
            
            # save best (by total loss)
            if L.item() < best_loss:
                best_loss = L.item()
                torch.save({
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "step": global_step,
                    "epoch": epoch
                }, file_identifier_out + '.pt')

            global_step += 1
    
    print("Done. Best loss:", best_loss)