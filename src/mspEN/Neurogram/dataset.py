import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import compute_class_weight

class NeurogramDataset(Dataset):
    """
    Windows neurograms and phoneme IDs into (F, T) tensors and soft 40-D labels.
    - neurogram: np.ndarray [N_time, F]
    - phonemes:  np.ndarray [N_time] with ints in {-1, 0..39}
    - T: window length (e.g., 512)
    - hop: step between windows (e.g., 512 for non-overlap, <512 for overlap)
    - min_labeled_ratio: minimum fraction of frames (within the window) that are labeled (!= -1)
    - normalize: if True, min-max normalize each window to [0,1] (per window); or pass a callable
    """
    def __init__(
        self,
        neurogram: np.ndarray,          # [N_time, F]
        phonemes: np.ndarray,           # [N_time] ints in {-1,0..39}
        T: int = 512,
        hop: int = 512,
        n_attrs: int = 40,
        min_labeled_ratio: float = 0.6,
        normalize=True,
        smooth_alpha: float = 0.0
    ):
        assert neurogram.ndim == 2, "neurogram must be [N_time, F]"
        assert phonemes.ndim == 1, "phonemes must be [N_time]"
        assert neurogram.shape[0] == phonemes.shape[0], "time alignment mismatch"

        self.neurogram = neurogram.astype(np.float32, copy=False)
        self.phonemes = phonemes.astype(np.int64, copy=False)
        self.T = int(T)
        self.hop = int(hop)
        self.n_attrs = int(n_attrs)
        self.min_labeled_ratio = float(min_labeled_ratio)
        self.normalize = normalize
        self.smooth_alpha = float(smooth_alpha)

        # --- indices of valid windows
        self.indices = []
        N = self.neurogram.shape[0]
        for start in range(0, N - T + 1, self.hop):
            end = start + T
            p_win = self.phonemes[start:end]
            labeled = (p_win != -1).sum()
            if labeled / float(T) >= self.min_labeled_ratio and labeled > 0:
                self.indices.append((start, end))

        # --- global class weights over ALL labeled frames (ignoring -1)
        valid = self.phonemes[self.phonemes != -1]
        if valid.size == 0:
            # avoid crash; default to ones
            self.class_weights_np = np.ones(self.n_attrs, dtype=np.float32)
        else:
            w = compute_class_weight(
                class_weight='balanced',
                classes=np.arange(self.n_attrs),
                y=valid
            ).astype(np.float32)
            self.class_weights_np = w


    def __len__(self):
        return len(self.indices)

    def _normalize_window(self, x_win: np.ndarray) -> np.ndarray:
        if callable(self.normalize):
            return self.normalize(x_win)
        if self.normalize is True:
            xmin = x_win.min()
            xmax = x_win.max()
            den = max(xmax - xmin, 1e-6)
            return (x_win - xmin) / den
        return x_win

    def _soft_label(self, p_win: np.ndarray) -> np.ndarray:
        mask = (p_win != -1)
        if not mask.any():
            return np.zeros(self.n_attrs, dtype=np.float32)
        ids = p_win[mask]
        counts = np.bincount(ids, minlength=self.n_attrs).astype(np.float32)
        label = counts / float(mask.sum())
        if self.smooth_alpha > 0.0:
            label = (1 - self.smooth_alpha) * label + self.smooth_alpha / self.n_attrs
        return label

    def __getitem__(self, idx: int):
        start, end = self.indices[idx]
        x_win = self.neurogram[start:end]           # [T, F]
        p_win = self.phonemes[start:end]            # [T]
        x_win = self._normalize_window(x_win)       # -> [0,1] typically
        label = self._soft_label(p_win)             # [40]

        # reshape (F, T) for Conv1d(bands=channels)
        x = torch.from_numpy(x_win.T.copy())        # (F, T)
        y = torch.from_numpy(label)                 # (40,)
        return x, y

    def class_weights(self, device=None) -> torch.Tensor:
        t = torch.from_numpy(self.class_weights_np)
        return t.to(device) if device is not None else t


def make_loader(
    neurogram_np,
    phonemes_np,
    batch_size=8,
    T=512,
    hop=512,
    min_labeled_ratio=0.6,
    shuffle=True,
    num_workers=0,
    smooth_alpha=0.0
):
    ds = NeurogramDataset(
        neurogram_np, phonemes_np,
        T=T, hop=hop, min_labeled_ratio=min_labeled_ratio,
        normalize=True, smooth_alpha=smooth_alpha
    )

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        drop_last=False, num_workers=num_workers
    )
    
    class_w = ds.class_weights()
    return ds, loader, class_w