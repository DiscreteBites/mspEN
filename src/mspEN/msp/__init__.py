import torch
import torch.nn as nn
import torch.nn.functional as F

class MSP(nn.Module):
    def __init__(self, label_dim: int = 40, latent_dim: int = 1024, encoding_type: str = 'one_hot', binary_c: int = 1):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.label_dim = label_dim
        self.encoding_type = encoding_type
        self.binary_c = binary_c
        self.M = nn.Parameter(torch.empty(self.label_dim, self.latent_dim))
        
        nn.init.xavier_normal_(self.M, gain=0.01)
        
    def encode_labels(self, label: torch.Tensor) -> torch.Tensor:
        """Convert phoneme indices [0..C-1] to desired encoding format"""

        # {0, ..., 39} -> {0, 1}^k
        if self.encoding_type == 'one_hot':
            return F.one_hot(label, num_classes=self.label_dim).float()

        # {-1, +1} version of one-hot: +1 for true class, -1 elsewhere
        elif self.encoding_type == 'binary':
            oh = F.one_hot(label, num_classes=self.label_dim).float()
            return oh * 2 - 1  # now in {-1, +1}
        
        # {-c, +c} version of one-hot: +c for true class, -c elsewhere
        elif self.encoding_type == 'binary_cc':
            oh = F.one_hot(label, num_classes=self.label_dim).float()
            return (oh * 2 - 1) * self.binary_c

        else:
            raise ValueError(f"Unknown encoding_type: {self.encoding_type}")
    
    def _loss_msp(self, label: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        encoded_label = self.encode_labels(label)
        L0 = F.mse_loss((z @ self.M.t()).view(-1), encoded_label.view(-1), reduction="none").sum()
        L1 = F.mse_loss((encoded_label @ self.M).view(-1), z.view(-1), reduction="none").sum()
        return L0 + L1

    def _loss_weight(self, target: torch.Tensor, label: torch.Tensor, z: torch.Tensor) -> float:
        encoded_label = self.encode_labels(label)
        return target.numel() / (encoded_label.numel() + z.numel())
    
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