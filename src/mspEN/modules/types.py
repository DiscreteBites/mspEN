from typing import Callable, Optional, Tuple, cast
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from pathlib import Path
Tensor = torch.Tensor
from dataclasses import dataclass

# Visualization function: (model, data, labels, output_dir, epoch, device) -> None
VisualizeFnType = Callable[[nn.Module, Tensor, Optional[Tensor], Path, int, torch.device], None]

@dataclass
class TrainingConfig:
    """Typed configuration for training"""
    use_adaptive_norm: bool = True
    num_epochs: int = 100
    scheduler_monitor_loss: str = 'Total'
    # learning_rate: float = 1e-4
    # weight_decay: float = 1e-5
    # beta: float = 0.7
    
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
    def loss(self, predict: Tensor, orig: Tensor, mu: Tensor, logvar: Tensor) -> Tuple[Tensor, float, float]:
        '''Returns: KLD loss and recon loss'''


DiscrOutType = Tensor
class DiscrType(nn.Module, ABC):
    def __call__(self, x: Tensor) -> DiscrOutType:
        return cast(DiscrOutType, super().__call__(x))   

    @abstractmethod
    def loss(self, prod: Tensor, target: Tensor, train_discriminator: bool = True) -> Tensor:
        '''Returns: Loss tensor, recon loss value, KLD loss value'''
