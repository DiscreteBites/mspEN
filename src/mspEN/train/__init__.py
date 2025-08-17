from typing import Optional, Tuple, Any, cast
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
import json

from mspEN.modules.types import VisualizeFnType, TrainingConfig, VAEType, DiscrType
from mspEN.msp import MSP

class mspVAETrainer:    
    def __init__(self,
        config: TrainingConfig,
        model_dir: Path,
        vae_model: VAEType, vae_optimizer: optim.Optimizer,
        msp: Optional[MSP] = None,
        discriminators: dict[str, Tuple[DiscrType, optim.Optimizer]] = {},
        visualize_fn: Optional[VisualizeFnType] = None, 
        device: Optional[torch.device] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        scheduler_monitor_loss: str = 'Total'
    ):
        
        # ==================================
        # Initialise internals
        # ==================================

        # Pull configs
        self.model_dir = model_dir
        self.config = config
        self.visualize_fn = visualize_fn

        # Setup models
        self.vae_model = vae_model
        self.vae_optimizer = vae_optimizer
        self.msp = msp
        self.discriminators = discriminators
        self.scheduler = scheduler
        self.scheduler_monitor_loss = scheduler_monitor_loss

        # Device setup
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vae_model.to(self.device)
        if self.msp is not None:
            self.msp.to(self.device)
        for name, (discr, opti) in self.discriminators.items():
            discr.to(self.device)
        
        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
        self.train_losses: dict[str, list[float]] = {}
        self.val_losses: dict[str, list[float]] = {}
        
        # Initialize loss store
        self.loss_keys = ['Total', 'Recon', 'VAE']
        if self.msp is not None:
            self.loss_keys.append('MSP')
        for name, _ in self.discriminators.items():
            self.loss_keys.append(f'Discriminator: {name}')

        assert self.scheduler_monitor_loss in self.loss_keys, "Scheduling on unknown loss"

        # Initialize loss tracking dictionaries
        for key in self.loss_keys:
            self.train_losses[key] = []
            self.val_losses[key] = []
          
        # Save config
        config_dict = vars(config) if hasattr(config, '__dict__') else config
        with open(self.model_dir / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        # ==================================
        # Log status message
        # ==================================

        print(f"MSP VAE Trainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  VAE parameters: {sum(p.numel() for p in self.vae_model.parameters()):,}")
        if self.discriminators:
            for name, (disc, _) in self.discriminators.items():
                print(f"  Discriminator '{name}' parameters: {sum(p.numel() for p in disc.parameters()):,}")
        print(f"  MSP enabled: {self.msp is not None}")
        print(f"  Output directory: {self.model_dir}")
    
    def train_epoch(self, train_loader: DataLoader[Tuple[Any, Any]]):
        """Train for one epoch"""
        self.vae_model.train()
        for name, (discr, opti) in self.discriminators.items():
            discr.train()

        # =========================
        # Setup epoch loss tracking
        # =========================

        epoch_loss: dict[str, float] = {k: 0.0 for k in self.loss_keys}
        
        # =====================
        # Epoch loop
        # =====================

        pbar = tqdm(train_loader, desc=f'Epoch {self.epoch+1}/{getattr(self.config, "num_epochs", "?")}')
        for batch_idx, (data, label) in enumerate(pbar):
            data = data.to(self.device)
            label = label.to(self.device)
            batch_size = data.size(0)
            
            # =====================
            # Forward pass of VAE
            # =====================

            self.vae_optimizer.zero_grad()
            prod, z, mu, logvar = self.vae_model(data)
            
            # =========================
            # Train ALL Discriminators
            # =========================

            for name, (discr, opti) in self.discriminators.items():
                opti.zero_grad()

                real = discr(data)
                fake = discr(prod.detach())
                loss_real = discr.loss(real, torch.ones_like(real))
                loss_fake = discr.loss(fake, torch.zeros_like(fake))  # Fixed: should be zeros for fake
                discr_loss = loss_real + loss_fake
                discr_loss.sum().backward()

                # Add gradient clipping for discriminators
                if hasattr(self.config, 'disc_grad_clip_norm') and self.config.disc_grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(discr.parameters(), self.config.disc_grad_clip_norm)
                
                opti.step()

            # ======================
            # Train VAE
            # ======================

            vae_loss, l_rec, l_vae = self.vae_model.loss(prod, data, mu, logvar)

            # Get MSP loss
            l_msp = torch.zeros((), device=self.device)
            if self.msp is not None:
                l_msp = self.msp.loss(data, label, z)
                
            # Get discriminator loss for generator
            Loss_discr = torch.zeros((), device=self.device)
            for name, (discr, opti) in self.discriminators.items():
                fake = discr(prod)
                this_loss = discr.loss(fake, torch.ones_like(fake), False).sum()
                epoch_loss[f'Discriminator: {name}'] += this_loss.item()
                Loss_discr = Loss_discr + this_loss

            # Train step
            total_loss = vae_loss + l_msp + Loss_discr
            total_loss.backward()
            
            # Add gradient clipping for VAE
            if hasattr(self.config, 'grad_clip_norm') and self.config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.vae_model.parameters(), self.config.grad_clip_norm)
            
            self.vae_optimizer.step()

            # ======================
            # Log epoch outs
            # ======================

            # Track epoch losses
            epoch_loss['Total'] += total_loss.item()
            epoch_loss['Recon'] += l_rec
            epoch_loss['VAE'] += l_vae

            if self.msp is not None:
                epoch_loss['MSP'] += l_msp.item()
            
            # Update progress bar
            status_dict = {
                'Loss': f'{total_loss.item() / batch_size:.4f}',
                'Recon': f'{l_rec / batch_size:.4f}',
                'KL': f'{l_vae / batch_size:.4f}',
                'LR': f'{self.vae_optimizer.param_groups[0]["lr"]:.6f}'
            }
            
            if self.msp is not None:
                status_dict['MSP Loss'] = f'{l_msp.item() / batch_size:.4f}'
            
            pbar.set_postfix(status_dict)
        
        # Calculate average losses
        num_batches = len(train_loader)
        for key in epoch_loss:
            epoch_loss[key] /= num_batches
        
        return epoch_loss

    def validate_epoch(self, val_loader: DataLoader[Tuple[Any, Any]]):
        """Validate for one epoch"""
        self.vae_model.eval()
        for _, (discr, _) in self.discriminators.items():
            discr.eval()
        
        # Setup validation loss tracking
        val_epoch_loss: dict[str, float] = {k: 0.0 for k in self.loss_keys}
        
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(tqdm(val_loader, desc='Validation')):                    
                data = data.to(self.device)
                label = label.to(self.device)

                # =====================
                # Forward pass of VAE
                # =====================
                prod, z, mu, logvar = self.vae_model(data)
                
                # Calculate VAE losses
                vae_loss, l_rec, l_vae = self.vae_model.loss(prod, data, mu, logvar)

                # Get MSP loss
                l_msp = torch.zeros((), device=self.device)
                if self.msp is not None:
                    l_msp = self.msp.loss(data, label, z)

                # Get discriminator losses
                Loss_discr = torch.zeros((), device=self.device)
                for name, (discr, _) in self.discriminators.items():
                    fake = discr(prod)
                    this_loss = discr.loss(fake, torch.ones_like(fake), False).sum()
                    val_epoch_loss[f'Discriminator: {name}'] += this_loss.item()
                    Loss_discr = Loss_discr + this_loss

                # Total loss
                total_loss = vae_loss + l_msp + Loss_discr

                # Track validation losses
                val_epoch_loss['Total'] += total_loss.item()
                val_epoch_loss['Recon'] += l_rec.item() if isinstance(l_rec, torch.Tensor) else l_rec
                val_epoch_loss['VAE'] += l_vae.item() if isinstance(l_vae, torch.Tensor) else l_vae

                if self.msp is not None:
                    val_epoch_loss['MSP'] += l_msp.item()
                
        # Calculate average losses
        num_batches = len(val_loader)
        for key in val_epoch_loss:
            val_epoch_loss[key] /= num_batches
        
        return val_epoch_loss
    
    def save_checkpoint(self, is_best: bool = False, extra_data: Optional[dict] = None):
        """Save model checkpoint (config saved separately as JSON)"""

        # ===================================
        # Create and save checkpoint object
        # ===================================
        checkpoint = {
            'epoch': self.epoch,
            'vae_model_state_dict': self.vae_model.state_dict(),
            'vae_optimizer_state_dict': self.vae_optimizer.state_dict(),
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        # Add discriminator states
        if self.discriminators:
            checkpoint['discriminator_state_dicts'] = {}
            checkpoint['discriminator_optimizer_state_dicts'] = {}
            for name, (discr, opti) in self.discriminators.items():
                checkpoint['discriminator_state_dicts'][name] = discr.state_dict()
                checkpoint['discriminator_optimizer_state_dicts'][name] = opti.state_dict()
        
        # Add scheduler state
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Add MSP state
        if self.msp is not None:
            checkpoint['msp_state_dict'] = self.msp.state_dict()
        
        # Add any extra data
        if extra_data:
            checkpoint.update(extra_data)

        # Save latest checkpoint (now weights_only=True compatible)
        torch.save(checkpoint, self.model_dir / 'latest_checkpoint.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.model_dir / 'best_checkpoint.pth')

    def load_checkpoint(self, checkpoint_path: Path, load_optimizer: bool = True, load_scheduler: bool = True):
        """Load model checkpoint using safe context manager"""
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # ==================================
        # Enforce old configs when resuming
        # ==================================

        config_path = checkpoint_path.parent / 'config.json'
        saved_config = json.load(open(config_path)) if config_path.exists() else {}
        current_config = vars(self.config)
        assert saved_config == current_config, f"Config mismatch: saved {saved_config} ≠ current {current_config}"

        # ==================================
        # Load model parameters
        # ==================================

        self.vae_model.load_state_dict(checkpoint['vae_model_state_dict'])
        
        if load_optimizer and 'vae_optimizer_state_dict' in checkpoint:
            self.vae_optimizer.load_state_dict(checkpoint['vae_optimizer_state_dict'])
        
        # Load discriminator states
        if self.discriminators and 'discriminator_state_dicts' in checkpoint:
            for name, (discr, opti) in self.discriminators.items():
                if name in checkpoint['discriminator_state_dicts']:
                    discr.load_state_dict(checkpoint['discriminator_state_dicts'][name])
                if load_optimizer and name in checkpoint.get('discriminator_optimizer_state_dicts', {}):
                    opti.load_state_dict(checkpoint['discriminator_optimizer_state_dicts'][name])
        
        if load_scheduler and self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load MSP state
        if self.msp and 'msp_state_dict' in checkpoint:
            self.msp.load_state_dict(checkpoint['msp_state_dict'])

        # ==================================
        # Load loss / epoch history
        # ==================================

        self.epoch = checkpoint.get('epoch', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.train_losses = checkpoint.get('train_losses', {k: [] for k in self.loss_keys})
        self.val_losses = checkpoint.get('val_losses', {k: [] for k in self.loss_keys})
        
        return checkpoint

    def visualize_reconstructions(self, val_loader: DataLoader, num_samples: int = 8):
        if self.visualize_fn is None:
            print("No visualization function provided. Skipping visualization.")
            return
        
        data, label = next(iter(val_loader))        
        
        self.visualize_fn(
            self.vae_model, data[:num_samples], label[:num_samples],
            self.model_dir, self.epoch, self.device 
        )
        
    def plot_losses(self):
        """Plot training and validation losses"""
        if not any(self.train_losses.values()):
            return
        
        # Get the number of epochs from any loss that has data
        epochs = None
        for key, losses in self.train_losses.items():
            if losses:
                epochs = range(1, len(losses) + 1)
                break
        
        if epochs is None:
            return
        
        # Create subplots based on number of loss components
        num_plots = len(self.loss_keys)
        cols = 3
        rows = (num_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        plot_idx = 0
        for key in self.loss_keys:
            row = plot_idx // cols
            col = plot_idx % cols
            
            if self.train_losses[key]:
                axes[row, col].plot(epochs, self.train_losses[key], 'b-', label='Training')
            if key in self.val_losses and self.val_losses[key]:
                axes[row, col].plot(epochs, self.val_losses[key], 'r-', label='Validation')
            
            axes[row, col].set_xlabel('Epoch')
            axes[row, col].set_ylabel(f'{key} Loss')
            axes[row, col].set_title(f'{key} Loss')
            axes[row, col].legend()
            axes[row, col].grid(True)
            
            plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.model_dir / f'training_losses_epoch_{self.epoch}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, resume_from: Optional[Path] = None):
        """Main training loop"""

        # ==================================
        # Pull checkpoint state if resuming
        # ==================================
        
        if resume_from:
            print(f"Resuming training from {resume_from}")
            self.load_checkpoint(resume_from)
            start_epoch = self.epoch + 1
        else:
            start_epoch = 0
        
        num_epochs = getattr(self.config, 'num_epochs', 100)
        print(f"Starting training for {num_epochs} epochs (from epoch {start_epoch})")
        
        for epoch in range(start_epoch, num_epochs):
            self.epoch = epoch
            
            # ==================================
            # Train one epoch
            # ==================================

            train_epoch_losses = self.train_epoch(train_loader)
            
            # ==================================
            # Validate one epoch
            # ==================================

            val_epoch_losses = {}
            if val_loader is not None:
                val_epoch_losses = self.validate_epoch(val_loader)
            
            # ==================================
            # Scheduler update
            # ==================================

            if self.scheduler:
                if hasattr(self.scheduler, 'step'):
                      if 'ReduceLROnPlateau' in str(type(self.scheduler)):
                        loss_type = self.scheduler_monitor_loss
                        monitor_loss = val_epoch_losses.get(loss_type, train_epoch_losses[loss_type])
                        cast(optim.lr_scheduler.ReduceLROnPlateau, self.scheduler).step(monitor_loss)

            # ==================================
            # Logging and storing losses
            # ==================================

            for key in self.loss_keys:
                self.train_losses[key].append(train_epoch_losses[key])
                if val_loader and key in val_epoch_losses:
                    self.val_losses[key].append(val_epoch_losses[key])
            
            # ==================================
            # Save checkpoint
            # ==================================

            current_loss = val_epoch_losses.get('Total', train_epoch_losses['Total'])
            is_best = current_loss < self.best_loss
            if is_best:
                self.best_loss = current_loss
            
            self.save_checkpoint(is_best)
            
            # ==================================
            # Std out prints
            # ==================================

            print(f'\nEpoch {epoch+1}/{num_epochs}:')
            print(f'  Train - Total: {train_epoch_losses["Total"]:.4f}, Recon: {train_epoch_losses["Recon"]:.4f}, VAE: {train_epoch_losses["VAE"]:.4f}')
            if self.msp is not None:
                print(f'  Train - MSP: {train_epoch_losses["MSP"]:.4f}')
            for name in self.discriminators.keys():
                disc_key = f'Discriminator: {name}'
                if disc_key in train_epoch_losses:
                    print(f'  Train - {disc_key}: {train_epoch_losses[disc_key]:.4f}')
            
            if val_loader:
                print(f'  Val - Total: {val_epoch_losses["Total"]:.4f}, Recon: {val_epoch_losses["Recon"]:.4f}, VAE: {val_epoch_losses["VAE"]:.4f}')
                if self.msp is not None:
                    print(f'  Val - MSP: {val_epoch_losses["MSP"]:.4f}')
                for name in self.discriminators.keys():
                    disc_key = f'Discriminator: {name}'
                    if disc_key in val_epoch_losses:
                        print(f'  Val - {disc_key}: {val_epoch_losses[disc_key]:.4f}')
            
            print(f'  Best Loss: {self.best_loss:.4f}')
            
            # Visualize reconstructions periodically
            vis_interval = getattr(self.config, 'vis_interval', 10)
            if val_loader and (epoch + 1) % vis_interval == 0:
                self.visualize_reconstructions(val_loader)
            
            # Plot losses periodically
            plot_interval = getattr(self.config, 'plot_interval', 10)
            if (epoch + 1) % plot_interval == 0:
                self.plot_losses()
        
        print(f'\nTraining completed! Best loss: {self.best_loss:.4f}')
        print(f'Model saved to: {self.model_dir}')
        
        return self.model_dir

# exports
__all__ = ['mspVAETrainer']