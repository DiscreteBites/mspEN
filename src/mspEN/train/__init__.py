from typing import Optional, Tuple, Any, cast

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
import json

from mspEN.modules.hyperparameters import EarlyStopping
from mspEN.modules.types import VisualizeFnType, TrainingConfig, VAEType, DiscrType, LossType
from mspEN.modules.loss import AutomaticLossScaler
from mspEN.msp import MSP
from mspEN.utils import fmt_train_status, make_status_bar

class mspVAETrainer:    
    def __init__(self,
        config: TrainingConfig,
        model_dir: Path,
    
        vae_model: VAEType, main_optimizer: optim.Optimizer,
        msp: Optional[MSP] = None,
        discriminators: dict[str, Tuple[DiscrType, optim.Optimizer]] = {},

        addon_losses: dict[str, LossType] = {},
        loss_scaling: str | None = None,
        loss_scaling_kwargs: dict[str, Any] = {},
        
        visualize_fn: Optional[VisualizeFnType] = None, 
        device: Optional[torch.device] = None,

        scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
        scheduler_monitor_loss: str = 'Total',
        early_stop: Optional[EarlyStopping] = None
    ):
        
        # ==================================
        # Initialise internals
        # ==================================

        self.model_dir = model_dir
        self.config = config
        self.visualize_fn = visualize_fn
                
        # ==================================
        # Setup Models
        # ==================================

        self.vae_model = vae_model
        self.main_optimizer = main_optimizer
        self.msp = msp
        self.discriminators = discriminators

        # ==================================
        # Setup Device
        # ==================================

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vae_model.to(self.device)
        if self.msp is not None:
            self.msp.to(self.device)
        for name, (discr, opti) in self.discriminators.items():
            discr.to(self.device)

        # Verify parameters on device
        assert all(p.device.type == self.device.type for p in self.vae_model.parameters())
        if self.msp is not None:
            assert all(p.device.type == self.device.type for p in self.msp.parameters())
        for name, (discr, opti) in self.discriminators.items():
            assert all(p.device.type == self.device.type for p in discr.parameters())

        # ==================================
        # Setup losses
        # ==================================
        
        self.addon_losses = addon_losses
        self.loss_keys = ['Recon', 'VAE']
        self.discriminator_loss_keys = []
        if self.msp is not None:
            self.loss_keys.append('MSP')
        for name, _ in self.discriminators.items():
            self.loss_keys.append(f'D:{name}')
            self.discriminator_loss_keys.append(f'D_train:{name}')
        for name, _ in self.addon_losses.items():
            self.loss_keys.append(f'A:{name}')
        
        self.loss_scaler = AutomaticLossScaler(
            self.loss_keys, method = loss_scaling,
            **loss_scaling_kwargs
        )
        
        # ==================================
        # Setup hyperparameter control
        # ==================================
        
        self.scheduler = scheduler
        self.scheduler_monitor_loss = scheduler_monitor_loss
        self.early_stop = early_stop

        assert self.scheduler_monitor_loss  == 'Total' \
            or self.scheduler_monitor_loss in self.loss_keys, "Scheduling on unknown loss"

        # ==================================
        # Initialise training state 
        # ==================================

        self.epoch = 0
        self.best_loss = float('inf')
        self.train_losses: dict[str, list[float]] = {
            'Total': [],
            **{key : [] for key in self.loss_keys},
            **{key : [] for key in self.discriminator_loss_keys},
        }
        self.val_losses: dict[str, list[float]] = {
            'Total': [],
            **{key : [] for key in self.loss_keys}
        }

        # ==================================
        # Save config and log status message
        # ==================================

        config_dict = vars(config) if hasattr(config, '__dict__') else config
        with open(self.model_dir / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

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

        epoch_loss: dict[str, float] = {
            'Total': 0.0,
            **{key: 0.0 for key in self.loss_keys}
        }
        discriminator_loss: dict[str, float] = {key: 0.0 for key in self.discriminator_loss_keys}
        
        # =====================
        # Epoch loop
        # =====================

        pbar = tqdm(train_loader, desc=f'Epoch {self.epoch+1}/{getattr(self.config, "num_epochs", "?")}')
        statusbar = make_status_bar()

        for batch_idx, (data, label) in enumerate(pbar):
            data = data.to(self.device)
            label = label.to(self.device)
            batch_size = data.size(0)
            
            # =====================
            # Forward pass of VAE
            # =====================

            self.main_optimizer.zero_grad()
            prod, z, mu, logvar = self.vae_model(data)
            
            # =========================
            # Train ALL Discriminators
            # =========================

            for name, (discr, opti) in self.discriminators.items():
                opti.zero_grad()
                
                real = discr(data)
                loss_real = discr.loss(real, torch.ones_like(real))

                fake = discr(prod.detach())
                loss_fake = discr.loss(fake, torch.zeros_like(fake))

                discr_loss = (loss_real + loss_fake).sum()
                discr_loss.backward()
                
                # Add gradient clipping for discriminators
                if hasattr(self.config, 'disc_grad_clip_norm') and self.config.disc_grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(discr.parameters(), self.config.disc_grad_clip_norm)
                
                opti.step()
                discriminator_loss[f'D_train:{name}'] += discr_loss.item()
                
            # ======================
            # Train VAE
            # ======================
            
            # Setup raw loss for auto scaling
            raw_losses = {key: torch.zeros((), device=self.device) for key in self.loss_keys}

            # Get VAE reconstruction and KL losses
            raw_losses['Recon'], raw_losses['VAE'] = self.vae_model.loss(prod, data, mu, logvar)

            # Get MSP loss
            if self.msp is not None:
                raw_losses['MSP'] = self.msp.loss(data, label, z)
            
            # Get discriminator losses (fool the discriminator)
            for name, (discr, opti) in self.discriminators.items():
                fake = discr(prod)
                raw_losses[f'D:{name}'] = discr.loss(fake, torch.ones_like(fake), False).sum()
            
            # Get addon losses
            for name, loss_fn in self.addon_losses.items():
                raw_losses[f'A:{name}'] = loss_fn(prod, data)
            
            # Train step
            scaled_losses = self.loss_scaler(raw_losses)
            total_loss = torch.stack(list(scaled_losses.values())).sum()
            total_loss.backward()

            # Add gradient clipping for VAE
            if hasattr(self.config, 'grad_clip_norm') and self.config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.vae_model.parameters(), self.config.grad_clip_norm)
            
            # Optimizer step
            self.main_optimizer.step()

            # ======================
            # Track losses
            # ======================
            
            # Track epoch losses
            epoch_loss['Total'] += total_loss.item()
            for key in self.loss_keys:
                epoch_loss[key] += scaled_losses[key].item()
            
            # Update progress and status
            statusbar.set_description_str(
                fmt_train_status(total_loss, scaled_losses, batch_size, self.loss_keys)
            )
            pbar.set_postfix({'LR': f'{self.main_optimizer.param_groups[0]["lr"]:.6f}'})
            
        statusbar.close()
        # Calculate average losses
        num_batches = len(train_loader)
        for key in epoch_loss:
            epoch_loss[key] /= num_batches
        
        for key in discriminator_loss:
            discriminator_loss[key] /= num_batches
        
        return epoch_loss, discriminator_loss

    def validate_epoch(self, val_loader: DataLoader[Tuple[Any, Any]]):
        """Validate for one epoch"""
        self.vae_model.eval()
        for _, (discr, _) in self.discriminators.items():
            discr.eval()
        
        # Setup validation loss tracking
        val_epoch_loss: dict[str, float] = {
            'Total': 0.0,
            **{key: 0.0 for key in self.loss_keys}
        }
        
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(tqdm(val_loader, desc='Validation')):                    
                data = data.to(self.device)
                label = label.to(self.device)

                # =====================
                # Forward pass of VAE
                # =====================
                prod, z, mu, logvar = self.vae_model(data)
                
                # Setup raw loss for auto scaling (matching training)
                raw_losses = {key: torch.zeros((), device=self.device) for key in self.loss_keys}

                # Get VAE reconstruction and KL losses
                raw_losses['Recon'], raw_losses['VAE'] = self.vae_model.loss(prod, data, mu, logvar)

                # Get MSP loss
                if self.msp is not None:
                    raw_losses['MSP'] = self.msp.loss(data, label, z)

                # Get discriminator losses (fool the discriminator)
                for name, (discr, _) in self.discriminators.items():
                    fake_output = discr(prod)
                    raw_losses[f'D:{name}'] = discr.loss(fake_output, torch.ones_like(fake_output), False).sum()

                # Get addon losses
                for name, loss_fn in self.addon_losses.items():
                    raw_losses[f'A:{name}'] = loss_fn(prod, data)

                # Apply loss scaling
                scaled_losses = self.loss_scaler(raw_losses)
                total_loss = torch.stack(list(scaled_losses.values())).sum()
                
                # Track validation losses
                val_epoch_loss['Total'] += total_loss.item()
                for key in self.loss_keys:
                    val_epoch_loss[key] += scaled_losses[key].item()
                    
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
            'vae_optimizer_state_dict': self.main_optimizer.state_dict(),
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
        
        # Add early stop state
        if self.early_stop is not None:
            checkpoint['early_stop_state_dict'] = self.early_stop.state_dict()

        # Add scheduler state
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Add loss scaler state
        if self.loss_scaler is not None:
            checkpoint['loss_scaler_state_dict'] = self.loss_scaler.state_dict()
        
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

    def load_checkpoint(self, 
        checkpoint_path: Path, load_optimizer: bool = True, load_stopper: bool = True, load_scheduler: bool = True, load_loss_scaler: bool = True
    ):
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
            self.main_optimizer.load_state_dict(checkpoint['vae_optimizer_state_dict'])
        
        # Load discriminator states
        if self.discriminators and 'discriminator_state_dicts' in checkpoint:
            for name, (discr, opti) in self.discriminators.items():
                if name in checkpoint['discriminator_state_dicts']:
                    discr.load_state_dict(checkpoint['discriminator_state_dicts'][name])
                if load_optimizer and name in checkpoint.get('discriminator_optimizer_state_dicts', {}):
                    opti.load_state_dict(checkpoint['discriminator_optimizer_state_dicts'][name])

        # Load MSP states
        if self.msp and 'msp_state_dict' in checkpoint:
            self.msp.load_state_dict(checkpoint['msp_state_dict'])
        
        # ========================
        # Hyperparameter states
        # =======================

        if load_stopper and self.early_stop and 'early_stop_state_dict' in checkpoint:
            self.early_stop.load_state_dict(checkpoint['early_stop_state_dict'])

        if load_scheduler and self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if load_loss_scaler and self.loss_scaler and 'loss_scaler_state_dict' in checkpoint:
            self.loss_scaler.load_state_dict(checkpoint['loss_scaler_state_dict'])
        
        # ==================================
        # Load loss / epoch history
        # ==================================

        self.epoch = checkpoint.get('epoch', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.train_losses = checkpoint.get('train_losses', {
            'Total': [],
            **{key : [] for key in self.loss_keys},
            **{key : [] for key in self.discriminator_loss_keys},
        })
        self.val_losses = checkpoint.get('val_losses', {
            'Total': [],
            **{key : [] for key in self.loss_keys}
        })
        
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
        num_plots = len(['Total', *self.loss_keys])
        cols = 3
        rows = (num_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        plot_idx = 0
        for key in ['Total', *self.loss_keys]:
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
        do_early_stop: bool = False

        for epoch in range(start_epoch, num_epochs):
            self.epoch = epoch
            
            # ================================
            # Update model hyperparameters
            # =================================

            self.vae_model.update_hyperparameters(epoch)
            if self.msp is not None:
                self.msp.update_hyperparameters(epoch)

            # ==================================
            # Train one epoch
            # ==================================

            train_epoch_losses, train_discriminator_losses = self.train_epoch(train_loader)
            
            # ==================================
            # Validate one epoch
            # ==================================
            
            val_epoch_losses = {}
            if val_loader is not None:
                val_epoch_losses = self.validate_epoch(val_loader)
            
            # ==================================
            # Scheduler update
            # ==================================

            if self.early_stop:
                loss_type = self.scheduler_monitor_loss
                monitor_loss = val_epoch_losses.get(loss_type, train_epoch_losses[loss_type])
                do_stop = self.early_stop(monitor_loss)
                if do_stop:
                    print(f"Early stopping triggered at epoch {epoch}")
                    print(f"Status: {self.early_stop.get_status()}")
                    do_early_stop = True

            if self.scheduler:
                if self.config.scheduler_type == 'plateau':
                    monitor_loss = val_epoch_losses.get(self.scheduler_monitor_loss, train_epoch_losses[self.scheduler_monitor_loss])
                    cast(optim.lr_scheduler.ReduceLROnPlateau, self.scheduler).step(monitor_loss)
                else:
                    self.scheduler.step()

            # ==================================
            # Logging and storing losses
            # ==================================

            for key in ['Total', *self.loss_keys]:
                self.train_losses[key].append(train_epoch_losses[key])
                if val_loader and key in val_epoch_losses:
                    self.val_losses[key].append(val_epoch_losses[key])

            for key in self.discriminator_loss_keys:
                self.train_losses[key].append(train_discriminator_losses[key])
            
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
            if val_loader and (do_early_stop or (epoch + 1) % vis_interval == 0):
                self.visualize_reconstructions(val_loader)
            
            # Plot losses periodically
            plot_interval = getattr(self.config, 'plot_interval', 10)
            if do_early_stop or (epoch + 1) % plot_interval == 0:
                self.plot_losses()

            if do_early_stop:
                break
        
        print(f'\nTraining completed! Best loss: {self.best_loss:.4f}')
        print(f'Model saved to: {self.model_dir}')
        
        return self.model_dir

# exports
__all__ = ['mspVAETrainer']