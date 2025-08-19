from typing import Iterable, Dict, Any

import torch
import torch.nn as nn

from mspEN.msp import MSP
from mspEN.modules import TrainingConfig

# ===========================
# Schedulers 
# ============================

def setup_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig):
    """Setup scheduler with plateau detection and early stopping"""
    if hasattr(config, 'scheduler_type') and config.scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.5,  # Reduce LR by half
            patience=10,  # Wait 10 epochs before reducing
            min_lr=1e-6
        )
    elif hasattr(config, 'scheduler_type') and config.scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.num_epochs,
            eta_min=1e-6
        )
    return None

# ===========================
# Early Stopping
# ============================

class EarlyStopping:
    """
    Early stopping utility with state persistence for checkpoint/resume functionality.
    
    Monitors a metric and stops training when no improvement is seen for a given patience period.
    Includes state_dict/load_state_dict methods for seamless checkpoint integration.
    """
    
    def __init__(self, 
        patience: int = 15, min_delta: float = 1e-4, mode: str = 'min'
    ):
        """
        Initialize EarlyStopping
        
        Args:
            patience: Number of epochs to wait for improvement before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' for loss (lower is better), 'max' for accuracy (higher is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        # Training state - these get saved/loaded
        self.counter = 0
        self.best_score = None
        self.early_stop_triggered = False
        
        # Set comparison function based on mode
        if mode == 'min':
            self.is_better = lambda current, best: current < best - self.min_delta
            self.best_score = float('inf')
        elif mode == 'max':
            self.is_better = lambda current, best: current > best + self.min_delta
            self.best_score = float('-inf')
        else:
            raise ValueError(f"Mode {mode} not supported. Use 'min' or 'max'")
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop based on current score
        
        Args:
            score: Current epoch's monitored metric value
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.early_stop_triggered:
            return True
            
        if self.is_better(score, self.best_score):
            # Improvement found
            self.best_score = score
            self.counter = 0
            return False
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop_triggered = True
                return True
            return False
    
    def state_dict(self) -> dict:
        """
        Return the state of the EarlyStopping object for checkpointing
        
        Returns:
            dict: State dictionary containing all necessary state
        """
        return {
            'patience': self.patience,
            'min_delta': self.min_delta,
            'mode': self.mode,
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stop_triggered': self.early_stop_triggered
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load the state of the EarlyStopping object from checkpoint
        
        Args:
            state_dict: State dictionary from a previous checkpoint
        """
        # Load configuration (should match initialization, but be safe)
        self.patience = state_dict.get('patience', self.patience)
        self.min_delta = state_dict.get('min_delta', self.min_delta)
        self.mode = state_dict.get('mode', self.mode)
        
        # Load training state
        self.counter = state_dict.get('counter', 0)
        self.best_score = state_dict.get('best_score', 
                                        float('inf') if self.mode == 'min' else float('-inf'))
        self.early_stop_triggered = state_dict.get('early_stop_triggered', False)
        
        # Recreate comparison function in case mode changed
        if self.mode == 'min':
            self.is_better = lambda current, best: current < best - self.min_delta
        elif self.mode == 'max':
            self.is_better = lambda current, best: current > best + self.min_delta
        else:
            raise ValueError(f"Mode {self.mode} not supported. Use 'min' or 'max'")
    
    def reset(self) -> None:
        """
        Reset the early stopping state (useful for new training runs)
        """
        self.counter = 0
        self.early_stop_triggered = False
        if self.mode == 'min':
            self.best_score = float('inf')
        else:
            self.best_score = float('-inf')
    
    def get_status(self) -> dict:
        """
        Get current status information for logging
        
        Returns:
            dict: Status information including counter, best score, etc.
        """
        return {
            'counter': self.counter,
            'patience': self.patience,
            'best_score': self.best_score,
            'triggered': self.early_stop_triggered,
            'epochs_remaining': max(0, self.patience - self.counter)
        }
    
    def __repr__(self) -> str:
        """String representation for debugging"""
        status = self.get_status()
        return (f"EarlyStopping(patience={self.patience}, min_delta={self.min_delta}, "
                f"mode='{self.mode}', counter={status['counter']}, "
                f"best_score={status['best_score']:.6f}, triggered={status['triggered']})")

# ============================
# Optimizer parameter building
# =============================

def build_param(mods: list[nn.Module]) -> Iterable[torch.nn.Parameter]:
    return [p for mod in mods for p in mod.parameters()]

def build_param_group(mods: list[nn.Module], lr: float=1e-3, wd: float=1e-2, lr_msp: float=3e-3) -> Iterable[Dict[str, Any]]:
    decay, no_decay, msp_grp = [], [], []
    for mod in mods:
        for name, p in mod.named_parameters():
            if not p.requires_grad:
                continue
            in_msp = mod is MSP
            if in_msp:
                msp_grp.append(p)
            elif p.ndim == 1 or name.endswith('.bias') or 'norm' in name or 'bn' in name:
                no_decay.append(p)
            else:
                decay.append(p)
    
    groups = []
    if decay:    groups.append({'params': decay,    'lr': lr,     'weight_decay': wd})
    if no_decay: groups.append({'params': no_decay, 'lr': lr,     'weight_decay': 0.0})
    if msp_grp:  groups.append({'params': msp_grp,  'lr': lr_msp, 'weight_decay': 0.0})
    return groups
