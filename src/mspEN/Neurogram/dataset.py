from __future__ import annotations
from typing import Tuple, Sequence

import numpy as np
from numpy.typing import NDArray
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split

Indices = Sequence[Tuple[int, int]]

class NeurogramDataset(Dataset):
    """
    Windows neurograms and phoneme IDs into (F, T) tensors and soft 40-D labels.
    - neurogram: np.ndarray [N_time, F]
    - phoneme:  np.ndarray [N_time] with ints in {0..39}
    - stream_idx: np.ndarray with indices to use
    - time_dim: window length (e.g., 50)
    - hop: step between windows or None for centered windows
    """
    def __init__(
        self,
        neurogram:  NDArray[np.float32],          # [N_time, F]
        phoneme: NDArray[np.int8],           # [N_time] ints in {0..39}
        stream_idx: NDArray[np.int32] | None = None,

        n_attrs: int = 40,
        val_split: float = 0.2,
        time_dim: int = 50,
        hop: int | None = None
    ):
        assert len(set(phoneme)) <= n_attrs, "got more phoneme than expected"
        assert neurogram.ndim == 2, "neurogram must be [N_time, F]"
        assert np.all((neurogram >= 0) & (neurogram <= 1)), "neurograms are expected to be pre normalised"
        assert phoneme.ndim == 1, "phoneme must be [N_time]"
        assert not np.any(phoneme < 0), "unlabeled phoneme present"
        assert neurogram.shape[0] == phoneme.shape[0], "time alignment mismatch"

        self.neurogram = neurogram.astype(np.float32, copy=False)
        self.phoneme = phoneme.astype(np.int8, copy=False)
        self.stream_idx = stream_idx.astype(np.int32, copy=False) if stream_idx is not None else np.arange(phoneme.shape[0], dtype=np.int32)

        self.n_attrs = n_attrs
        self.val_split = val_split
        self.time_dim = int(time_dim)
        self.hop = int(hop) if hop is not None else None
        
        # Compute centered phoneme
        self.indices = self.get_centered_phonemes()
        self.train_indices, self.val_indices = self.train_val_split()
        
    def train_val_split(self, seed=42) -> Tuple[Subset, Subset]:
        labels = self.phoneme[[start for (start, _) in self.indices ]]
        all_pos = np.arange(len(self.indices))

        train, val = train_test_split(
            all_pos,
            test_size=self.val_split,
            stratify= labels,
            random_state=seed
    )   

        return Subset(self, train.tolist()), Subset(self, val.tolist())

    def get_centered_phonemes(self) -> Indices:
        '''Get centered stable phoneme regions
        '''

        # Append -1 to the end so that the last element is a boundary end
        deltas = np.ediff1d(self.phoneme[self.stream_idx], to_end=-1)
        boundary_ends = self.stream_idx[deltas != 0]
        boundary_starts = np.insert(boundary_ends[:-1]+1, 0, 0)

        lengths = boundary_ends - boundary_starts +1
        mask = lengths >= self.time_dim

        left_win = int(np.floor((self.time_dim -1) / 2))
        right_win = int(np.ceil((self.time_dim -1) / 2))
        indices = []

        if self.hop is None:
            # centered indices
            centers = np.floor((boundary_ends[mask] + boundary_starts[mask]) /2)
            indices = [(int(c -left_win), int(c +right_win)) for c in centers]
        else:
            # rolling windows with hop
            jumps = np.floor((lengths[mask] - self.time_dim) / self.hop).astype(int)
            offsets = np.concatenate([np.arange(j + 1) * self.hop for j in jumps])
            centers = np.repeat(boundary_starts[mask] + left_win, jumps + 1) + offsets

            indices = [(int(c -left_win), int(c +right_win)) for c in centers]

        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        start, end = self.indices[idx]
        x_win = self.neurogram[start:end+1] # [T, 150]
        phn_idx = self.phoneme[start]   # phoneme index

        x = torch.from_numpy(x_win.T.copy()).unsqueeze(0) # (1, 150, T)
        label = int(phn_idx)

        return x, label
        

def make_loader(
    neurogram_path: Path | str,
    phoneme_path: Path | str,
    stream_path: Path | str,
    
    n_attrs: int = 40,
    val_split: float = 0.2,
    time_dim: int =50,
    hop: int | None = None,

    batch_size: int =30,
    shuffle: bool =True,
    num_workers: int =0,
):
    neurogram_np = np.load( neurogram_path )
    phoneme_np = np.load( phoneme_path )
    stream_np = np.load( stream_path )
    
    ds = NeurogramDataset(
        neurogram_np, phoneme_np, stream_np,
        n_attrs=n_attrs, val_split=val_split,
        time_dim=time_dim, hop=hop
    )

    train_ds, val_ds = ds.train_val_split()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle,
        drop_last=False, num_workers=num_workers
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=shuffle,
        drop_last=False, num_workers=num_workers
    )

    return ds, train_loader, val_loader