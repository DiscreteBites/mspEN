import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

class MSP(nn.Module):
    class_weights: torch.Tensor

    def __init__(
        self, label_dim: int = 40, latent_dim: int = 1024, 
        label_type: str = 'pure', time_dim: int = 50,
        encoding_type: str = 'one_hot', binary_c: int = 1,
        class_weights: Optional[torch.Tensor] = None,
        beta_weight: float = 1
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.label_dim = label_dim
        self.label_type = label_type
        self.time_dim = time_dim
        self.encoding_type = encoding_type
        self.binary_c = binary_c
        self.beta_weight = beta_weight;

        self.M = nn.Parameter(torch.empty(self.label_dim, self.latent_dim))

        # Register class weights as buffer (moves with model to device)
        if class_weights is not None:
            assert len(class_weights) == label_dim, f"Expected {label_dim} weights, got {len(class_weights)}"
            self.register_buffer('class_weights', class_weights.clone())
        else:
            self.register_buffer('class_weights', torch.ones(label_dim))
        
        nn.init.xavier_normal_(self.M, gain=0.01)
        
    def encode_labels(self, label: torch.Tensor) -> torch.Tensor:
        assert label.dim() == 2, f"Labels should be 2D, got shape {label.shape}"
        assert label.size(1) == self.time_dim, f"Labels should have {self.time_dim} frames, got {label.size(1)}"
        assert label.dtype == torch.int8, f"Labels should be int8 type, got {label.dtype}"
        assert torch.all(label >= 0), f"Labels should be non-negative, got min value {label.min().item()}"
        assert torch.all(label < self.label_dim), f"Labels should be < {self.label_dim}, got max value {label.max().item()}"

        label = label.long()
        if self.label_type == 'pure':
            pure_label = label[:, 0]                
            assert torch.all(label == pure_label.unsqueeze(1)), f"Pure labels: all elements in each row should be identical"
            return self._encode_pure_label(pure_label)
        
        elif self.label_type == 'mixed':
            return self._encode_mixed_label(label)
        
        else:
            raise ValueError(f"Unknown label_type: {self.label_type}")
    
    def _encode_mixed_label(self, label: torch.Tensor):
        """Convert mixed phoneme indices [0..C-1] to desired encoding format"""

        batch_size, seq_len = label.shape
        proportions = torch.zeros(batch_size, self.label_dim, device=label.device, dtype=torch.float)
        
        # Compute proportions for each batch item
        for i in range(batch_size):
            unique_ids, counts = torch.unique(label[i], return_counts=True)
            proportions[i, unique_ids] = counts.float() / seq_len
        
         # {0, ..., 39} -> {0, 1}^k
        if self.encoding_type == 'one_hot':
            return proportions  # Already in [0,1] format
        
        # {-1, +1} version: scale [0,1] to [-1,+1]
        elif self.encoding_type == 'binary':
            return proportions * 2 - 1  # [0,1] -> [-1,+1]
        
        # {-c, +c} version: scale [0,1] to [-c,+c]
        elif self.encoding_type == 'binary_cc':
            return (proportions * 2 - 1) * self.binary_c  # [0,1] -> [-c,+c]
        
        else:
            raise ValueError(f"Unknown encoding_type: {self.encoding_type}")
        
    def _encode_pure_label(self, pure_label: torch.Tensor) -> torch.Tensor:
        """Convert pure phoneme indices [0..C-1] to desired encoding format"""
        # {0, ..., 39} -> {0, 1}^k
        if self.encoding_type == 'one_hot':
            return F.one_hot(pure_label, num_classes=self.label_dim).float()

        # {-1, +1} version of one-hot: +1 for true class, -1 elsewhere
        elif self.encoding_type == 'binary':
            oh = F.one_hot(pure_label, num_classes=self.label_dim).float()
            return oh * 2 - 1  # now in {-1, +1}
        
        # {-c, +c} version of one-hot: +c for true class, -c elsewhere
        elif self.encoding_type == 'binary_cc':
            oh = F.one_hot(pure_label, num_classes=self.label_dim).float()
            return (oh * 2 - 1) * self.binary_c
        else:
            raise ValueError(f"Unknown encoding_type: {self.encoding_type}")
    
    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass implementing MSP projection"""
        encoded_labels = self.encode_labels(labels)
        y_hat = z @ self.M.t()  # Project latent to attribute space
        return y_hat, encoded_labels
    
    def _loss_msp(self, label: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        encoded_label = self.encode_labels(label)
        
        # L1: ||M * z - y||^2 (attribute prediction loss)
        L1_per_class = F.mse_loss((z @ self.M.t()).view(-1), encoded_label.view(-1), reduction="none").sum()
        
        # Weight L1 attribute information
        assert hasattr(self, 'class_weights'), "class_weights buffer not initialized"
        L1_weighted = (L1_per_class * self.class_weights.unsqueeze(0)).sum()
        
        # L2: ||z - M^T * y||^2 (minimize non-attribute information) 
        # Leave L2 unweighted
        L2 = F.mse_loss(z.view(-1), (encoded_label @ self.M).view(-1), reduction="none").sum()
        
        return L1_weighted + L2
    
    def update_hyperparameters(self, epoch):
        return
    
    def _loss_weight(self, target: torch.Tensor, label: torch.Tensor, z: torch.Tensor) -> float:
        encoded_label = self.encode_labels(label)
        return self.beta_weight * (target.numel() / (encoded_label.numel() + z.numel()))
    
    def loss(self, target: torch.Tensor, label: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self._loss_weight(target, label, z) * self._loss_msp(label, z)
    

    # def predict(self, x: torch.Tensor, new_ls=None, weight=0.0):
    #     z, _ = self.encode(x)
    #     if new_ls is not None:
    #         zl = z @ self.M.t()
    #         d = torch.zeros_like(zl)
    #         for i, v in new_ls:
    #             d[:, i] = v * weight - zl[:, i]
    #         z += d @ self.M
    #     prod = self.decoder(z)
    #     return prod
 
    # def predict_ex(self, x, encoded_label, new_ls=None, weight=0.0):
    #     return self.predict(x, new_ls, weight)

    # def get_U(self, eps=0e-5):
    #     from scipy import linalg, compress
    #     M = self.M.detach().cpu()
    #     A = torch.zeros(M.shape[0] - M.shape[0], M.shape[1])
    #     A = torch.cat([M, A])
    #     u, s, vh = linalg.svd(A.numpy())
    #     null_mask = (s <= eps)
    #     null_space = compress(null_mask, vh, axis=-1)
    #     N = torch.tensor(null_space)
    #     return torch.cat([self.M, N.to(self.M.device)]) 

__all__ = [
    'MSP'
]