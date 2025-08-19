import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Any
from collections import deque
import numpy as np

from mspEN.modules.types import LossDict, LossType

class AutomaticLossScaler:
    """Automatic loss scaling strategies for multi-loss training"""
    
    def __init__(self, 
        loss_keys: list[str],
        method: str | None ='magnitude_matching',
        target_proportions: Optional[dict[str, float]] = None,
        window_size: int=100
    ):
        self.loss_keys = loss_keys
        self.method = method
        self.window_size = window_size
        self.target_proportions = self._norm_proportions(target_proportions) if target_proportions is not None \
            else {key : 1/len(loss_keys) for key in loss_keys }
                
        # Running statistics for different methods
        self.loss_history: dict[str, deque[float]] = {}
        self.grad_norm_history: dict[str, deque[float]] = {}
        self.scales: dict[str, float] = {}
        self.update_count: int = 0

    @staticmethod
    def _norm_proportions(proportions: dict[str, float]):
        total = sum(proportions.values())
        return {k: v/total for k, v in proportions.items()}
    
    def __call__(self, losses_dict: LossDict, model_params=None) -> LossDict:
        """
        Update statistics and return scaled losses
        
        Args:
            losses_dict: {'loss_name': loss_tensor, ...}
            model_params: model parameters for gradient-based scaling
        
        Returns:
            scaled_losses_dict: {'loss_name': scaled_loss_tensor, ...}
        """
        self.update_count += 1
        
        if self.method is None:
            return losses_dict
        
        assert set(losses_dict.keys()) == set(self.loss_keys), "Mismatched losses in auto scaler"

        if self.method == 'magnitude_matching':
            return self._magnitude_matching(losses_dict)
        elif self.method == 'gradient_matching':
            return self._gradient_matching(losses_dict, model_params)
        elif self.method == 'moving_average':
            return self._moving_average_scaling(losses_dict)
        elif self.method == 'percentile_based':
            return self._percentile_based_scaling(losses_dict)
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")
    
    def _magnitude_matching(self, losses_dict: LossDict) -> LossDict:
        """Scale losses to have similar magnitudes"""
        # Convert to raw values for comparison
        loss_values = {name: loss.item() if torch.is_tensor(loss) else loss 
                      for name, loss in losses_dict.items()}
        
        # Find reference scale (e.g., reconstruction loss or largest loss)
        if 'Recon' in loss_values:
            reference_scale = loss_values['Recon']
        else:
            reference_scale = max(loss_values.values())
        
        if reference_scale == 0:
            reference_scale = 1.0
        
        scaled_losses = {}
        scales_used = {}
        
        for name, loss in losses_dict.items():
            current_scale = loss.item()

            if current_scale == 0:
                scale_factor = 1.0
            else:
                # Scale to match reference magnitude
                scale_factor = reference_scale / current_scale
                # Clamp to reasonable range
                scale_factor = torch.clamp(torch.tensor(scale_factor), 0.01, 100.0).item()
            
            proportional_sf = scale_factor * self.target_proportions[name]
            scaled_losses[name] = loss * proportional_sf 
            self.scales[name] = proportional_sf
        
        return scaled_losses
    
    def _gradient_matching(self, losses_dict: LossDict, model_params) -> LossDict:
        """Scale losses based on gradient magnitudes they produce"""
        if model_params is None:
            return self._magnitude_matching(losses_dict)  # Fallback
        
        # Compute gradient norms for each loss
        grad_norms = {}
        scaled_losses = {}
        
        # Reference loss (usually reconstruction)
        ref_loss_name = 'Recon' if 'Recon' in losses_dict else list(losses_dict.keys())[0]
        
        for name, loss in losses_dict.items():
            if loss.requires_grad:
                # Compute gradients
                grads = torch.autograd.grad(
                    loss, model_params, retain_graph=True, create_graph=False, allow_unused=True
                )
                # Calculate gradient norm
                grad_norm = 0.0
                for grad in grads:
                    if grad is not None:
                        grad_norm += grad.norm().item() ** 2
                grad_norm = grad_norm ** 0.5
                grad_norms[name] = grad_norm
            else:
                grad_norms[name] = 0.0
        
        # Scale relative to reference
        ref_grad_norm = grad_norms.get(ref_loss_name, 1.0)
        if ref_grad_norm == 0:
            ref_grad_norm = 1.0
        
        for name, loss in losses_dict.items():
            current_grad_norm = grad_norms[name]
            if current_grad_norm == 0:
                scale_factor = 1.0
            else:
                scale_factor = ref_grad_norm / current_grad_norm
                scale_factor = torch.clamp(torch.tensor(scale_factor), 0.01, 100.0).item()
            
            proportional_sf = scale_factor * self.target_proportions[name]
            scaled_losses[name] = loss * proportional_sf 
            self.scales[name] = proportional_sf
        
        return scaled_losses
    
    def _moving_average_scaling(self, losses_dict: LossDict) -> LossDict:
        """Scale based on moving averages of loss magnitudes"""
        # Update history
        for name, loss in losses_dict.items():
            if name not in self.loss_history:
                self.loss_history[name] = deque(maxlen=self.window_size)
            
            loss_val = loss.item()
            self.loss_history[name].append(loss_val)
        
        # Compute moving averages
        avg_losses = {}
        for name, history in self.loss_history.items():
            if len(history) > 0:
                avg_losses[name] = np.mean(history)
            else:
                avg_losses[name] = 1.0
        
        # Scale to target reference
        if 'Recon' in avg_losses:
            reference = avg_losses['Recon']
        else:
            reference = max(avg_losses.values())
        
        if reference == 0:
            reference = 1.0
        
        scaled_losses = {}
        for name, loss in losses_dict.items():
            avg_loss = avg_losses[name]
            if avg_loss == 0:
                scale_factor = 1.0
            else:
                scale_factor = reference / avg_loss
                scale_factor = torch.clamp(torch.tensor(scale_factor), 0.01, 100.0).item()
            
            proportional_sf = scale_factor * self.target_proportions[name]
            scaled_losses[name] = loss * proportional_sf 
            self.scales[name] = proportional_sf
        
        return scaled_losses
    
    def _percentile_based_scaling(self, losses_dict: LossDict) -> LossDict:
        """Scale based on percentiles of historical loss values"""
        # Update history
        for name, loss in losses_dict.items():
            if name not in self.loss_history:
                self.loss_history[name] = deque(maxlen=self.window_size)
            
            loss_val = loss.item()
            self.loss_history[name].append(loss_val)
        
        # Compute percentiles (e.g., 50th percentile = median)
        percentile_losses = {}
        for name, history in self.loss_history.items():
            if len(history) > 10:  # Need enough samples
                percentile_losses[name] = np.percentile(history, 50)
            else:
                percentile_losses[name] = np.mean(history) if len(history) > 0 else 1.0
        
        # Scale relative to reference
        if 'Recon' in percentile_losses:
            reference = percentile_losses['Recon']
        else:
            reference = max(percentile_losses.values())
        
        if reference == 0:
            reference = 1.0
        
        scaled_losses = {}
        for name, loss in losses_dict.items():
            percentile_loss = percentile_losses[name]
            if percentile_loss == 0:
                scale_factor = 1.0
            else:
                scale_factor = reference / percentile_loss
                scale_factor = torch.clamp(torch.tensor(scale_factor), 0.01, 100.0).item()
            
            proportional_sf = scale_factor * self.target_proportions[name]
            scaled_losses[name] = loss * proportional_sf 
            self.scales[name] = proportional_sf
        
        return scaled_losses
    
    def state_dict(self) -> dict[str, Any]:
        """Return the state dictionary for saving/loading"""
        return {
            'method': self.method,
            'window_size': self.window_size,
            'target_proportions': self.target_proportions,
            'loss_history': {name: list(history) for name, history in self.loss_history.items()},
            'grad_norm_history': {name: list(history) for name, history in self.grad_norm_history.items()},
            'scales': self.scales.copy(),
            'update_count': self.update_count
        }
    
    def load_state_dict(self, state_dict: dict[str, Any]):
        """Load state from a state dictionary"""
        self.method = state_dict.get('method', self.method)
        self.window_size = state_dict.get('window_size', self.window_size)
        self.target_proportions = state_dict.get('target_proportions', self.target_proportions)
        
        # Restore loss history as deques
        loss_history_data = state_dict.get('loss_history', {})
        self.loss_history = {}
        for name, history_list in loss_history_data.items():
            self.loss_history[name] = deque(history_list, maxlen=self.window_size)
        
        # Restore grad norm history as deques
        grad_norm_history_data = state_dict.get('grad_norm_history', {})
        self.grad_norm_history = {}
        for name, history_list in grad_norm_history_data.items():
            self.grad_norm_history[name] = deque(history_list, maxlen=self.window_size)
        
        self.scales = state_dict.get('scales', {}).copy()
        self.update_count = state_dict.get('update_count', 0)

class TemporalConsistencyLoss(LossType):
    """Encourage smooth reconstructions across time"""
    
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight
    
    def forward(self, reconstructed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        # First-order temporal difference
        orig_diff = original[..., 1:] - original[..., :-1]
        recon_diff = reconstructed[..., 1:] - reconstructed[..., :-1]
        
        temporal_loss = F.mse_loss(recon_diff, orig_diff)
        
        # Second-order difference (curvature)
        if original.shape[-1] > 2:
            orig_diff2 = orig_diff[..., 1:] - orig_diff[..., :-1]
            recon_diff2 = recon_diff[..., 1:] - recon_diff[..., :-1]
            curvature_loss = F.mse_loss(recon_diff2, orig_diff2)
            temporal_loss += 0.5 * curvature_loss
        
        return self.weight * temporal_loss

class SpectralLoss(LossType):
    """Preserve spectral characteristics important for ASR"""
    
    def __init__(self, weight=0.05):
        super().__init__()
        self.weight = weight
    
    def forward(self, reconstructed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        # FFT along time dimension
        orig_fft = torch.fft.fft(original, dim=-1)
        recon_fft = torch.fft.fft(reconstructed, dim=-1)
        
        # Compare magnitude and phase separately
        mag_loss = F.mse_loss(torch.abs(recon_fft), torch.abs(orig_fft))
        phase_loss = F.mse_loss(torch.angle(recon_fft), torch.angle(orig_fft))
        
        return self.weight * (mag_loss + 0.5 * phase_loss)