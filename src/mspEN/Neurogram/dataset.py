import numpy as np
from numpy.typing import NDArray

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import compute_class_weight

class NeurogramDataset(Dataset):
    """
    Windows neurograms and phoneme IDs into (F, T) tensors and soft 40-D labels.
    - neurogram: np.ndarray [N_time, F]
    - phonemes:  np.ndarray [N_time] with ints in {0..39}
    - T: window length (e.g., 512)
    - hop: step between windows (e.g., 512 for non-overlap, <512 for overlap)
    """
    def __init__(
        self,
        neurogram: np.ndarray,          # [N_time, F]
        phonemes: np.ndarray,           # [N_time] ints in {-1,0..39}
        T: int = 512,
        hop: int = 512,
        n_attrs: int = 40,
        smooth_alpha: float = 0.0,
        
        class_weight_mode: str | None = "balanced",     # "balanced"
		class_weight_normalize: str | None = "mean",
		class_weight_clip: float | None = None,
    ):
        assert neurogram.ndim == 2, "neurogram must be [N_time, F]"
        assert np.all((neurogram >= 0) & (neurogram <= 1)), "neurograms are expected to be pre normalised"
        assert phonemes.ndim == 1, "phonemes must be [N_time]"
        assert not np.any(phonemes < 0), "unlabeled phonemes present"
        assert neurogram.shape[0] == phonemes.shape[0], "time alignment mismatch"

        self.neurogram = neurogram.astype(np.float32, copy=False)
        self.phonemes = phonemes.astype(np.int64, copy=False)
        self.T = int(T)
        self.hop = int(hop)
        self.n_attrs = int(n_attrs)
        self.smooth_alpha = float(smooth_alpha)

        # --- indices of valid windows
        # window start/end indices (no labeled-ratio gate needed)
        self.indices = [(s, s + self.T) for s in range(0, len(self.phonemes) - self.T + 1, self.hop)]

        # --- global class weights over ALL labeled frames (ignoring -1)
        self.class_weights_np = self._compute_class_weights(
            mode=class_weight_mode,
			normalize=class_weight_normalize,
			clip=class_weight_clip
        )

    def _compute_class_weights(
        self, 
        mode: str | None, 
        normalize: str | None,
        clip: float | None
    ) -> NDArray[np.float32]:
    
        if mode == "balanced":
            w = compute_class_weight(
                class_weight="balanced",
                classes=np.arange(self.n_attrs),
                y=self.phonemes
            ).astype(np.float32)
        else:
            w = np.ones(self.n_attrs, dtype=np.float32)

        if normalize == "mean":
            m = float(w.mean()) if w.size else 1.0
            if m > 0:
                w = w / m
        
        if (clip is not None) and (clip > 0):
            w = np.minimum(w, float(clip))
        
        return w.astype(np.float32)


    def __len__(self):
        return len(self.indices)

    def _soft_label(self, p_win: NDArray) -> NDArray:
        # average one-hot over the window
        T = p_win.shape[0]
        counts = np.bincount(p_win, minlength=self.n_attrs).astype(np.float32)
        label = counts / float(T)

        if self.smooth_alpha > 0.0:
            label = (1 - self.smooth_alpha) * label + self.smooth_alpha / self.n_attrs
        
        return label

    def __getitem__(self, idx: int):
        start, end = self.indices[idx]
        x_win = self.neurogram[start:end]      # [T, F]
        p_win = self.phonemes[start:end]       # [T]

        label = self._soft_label(p_win)        # [40], floats in [0,1]    
        
        # reshape to (F, T) for Conv1d (bands=channels)
        x = torch.from_numpy(x_win.T.copy())   # (F, T)
        y = torch.from_numpy(label)            # (40,)

        return x, y

    def class_weights(self, device=None) -> torch.Tensor:
        t = torch.from_numpy(self.class_weights_np)
        return t.to(device) if device is not None else t


def make_loader(
    neurogram_np,
    phonemes_np,
    batch_size=20,
    T=128,
    hop=128/4,
    shuffle=True,
    num_workers=0,
    smooth_alpha=0.0,
    class_weight_mode: str | None= "balanced",
	class_weight_normalize: str | None = "mean",
	class_weight_clip: float | None = None,
):
    ds = NeurogramDataset(
        neurogram_np, phonemes_np,
        T=T, hop=hop, normalize=True, smooth_alpha=smooth_alpha,
        class_weight_mode=class_weight_mode,
        class_weight_normalize=class_weight_normalize,
        class_weight_clip=class_weight_clip,
    )

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        drop_last=False, num_workers=num_workers
    )
    
    return ds, loader