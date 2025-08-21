from typing import Callable, Any, Optional, Tuple, cast
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from pathlib import Path
Tensor = torch.Tensor
from dataclasses import dataclass
from collections.abc import Mapping

LossDict = Mapping[str, torch.Tensor]

# Visualization function: (model, data, labels, output_dir, epoch, device) -> None
VisualizeFnType = Callable[[nn.Module, Tensor, Tensor, Path, int, torch.device], None]

@dataclass
class TrainingConfig:
    """Typed configuration for training"""
    # Data parameters
    data_strategy: str
    beta_kld: float = 0.7
    label_type: str = 'mixed'
    
    # Model Parameters
    use_adaptive_norm: bool = True
    num_epochs: int = 100
    
    # Loss parameters
    loss_scaling: str | None = None
    
    # Scheduling parameters
    scheduler_type: str = 'plateau'
    scheduler_monitor_loss: str = 'Total'
    early_stopping: bool = True

    # Logging parameters
    log_interval: int = 20
    vis_interval: int = 20
    plot_interval: int = 20
     
    # Gradient clipping
    grad_clip_norm: float = 1.0
    disc_grad_clip_norm: float = 1.0

VAEOutType = Tuple[Tensor, Tensor, Tensor, Tensor]
class VAEType(nn.Module, ABC):
    def __call__(self, x: Tensor) -> VAEOutType:
        return cast(VAEOutType, super().__call__(x))
    
    @abstractmethod
    def encode(self, x: Tensor) -> torch.Tensor:
        '''Encode data into latent representation'''

    @abstractmethod
    def decode(self, z: Tensor) -> torch.Tensor:
        '''Decode data back'''

    @abstractmethod
    def update_hyperparameters(self, epoch: int) -> None:
        '''Schedule hyperparameters'''
    
    @abstractmethod
    def loss(self, predict: Tensor, orig: Tensor, mu: Tensor, logvar: Tensor) -> Tuple[Tensor, Tensor]:
        '''Returns: Recon loss and beta KLD loss'''

DiscrOutType = Tensor
class DiscrType(nn.Module, ABC):
    def __call__(self, x: Tensor) -> DiscrOutType:
        return cast(DiscrOutType, super().__call__(x))   
    
    @abstractmethod
    def loss(self, pred: Tensor, target: Tensor, train_discriminator: bool = True) -> Tensor:
        '''Returns: Loss tensor, recon loss value, KLD loss value'''

class LossType(nn.Module, ABC):
    def __call__(self, reconstructed: Tensor, original: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(reconstructed, original))
