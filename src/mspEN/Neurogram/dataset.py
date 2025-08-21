from __future__ import annotations
from typing import Tuple, Sequence, Optional

import numpy as np
from numpy.typing import NDArray
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, Subset
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
        neurogram:  NDArray[np.float32],
        phoneme: NDArray[np.int8],
        stream_idx: NDArray[np.int32] | None = None,
        
        strategy: str = 'centered',

        n_attrs: int = 40,
        val_split: float = 0.2,
        time_dim: int = 100,
        hop: int | None = None,
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
        
        # Compute phoneme windows
        if strategy == 'naive':
            self.indices = self._get_naive_window()
        elif strategy == 'naive_rolling':
            self.indices = self._get_rolling_window()
        elif strategy == 'centered':
            self.hop = None
            self.indices = self._get_centered_window()
        elif strategy == 'centered_rolling':
            assert self.hop is not None, "centered rolling requires a hop size"
            self.indices = self._get_centered_window()
        else:
            raise ValueError(f'Unknown windowing strategy {strategy}')
        
        self.train_indices, self.val_indices = self.train_val_split()
    
    def train_val_split(self, seed=42) -> Tuple[Optional[Subset], Optional[Subset]]:
        labels = self.phoneme[[start for (start, _) in self.indices ]]
        all_pos = np.arange(len(self.indices))
        
        if self.val_split == 0:
            return Subset(self, all_pos.tolist()), None
        
        if self.val_split == 1:
            return None, Subset(self, all_pos.tolist())
        
        train, val = train_test_split(
            all_pos,
            test_size=self.val_split,
            stratify= labels,
            random_state=seed
        )      

        return Subset(self, train.tolist()), Subset(self, val.tolist())

    def _get_naive_window(self) -> Indices:
        self.hop = self.time_dim
        step = int(self.hop)
        
        length = self.neurogram.shape[0]
        max_start = length - self.time_dim
        
        if max_start < 0:
            return []  # Not enough data for even one window
        
        return [(i, i + self.time_dim - 1) for i in range(0, max_start + 1, step)]  
    
    def _get_rolling_window(self) -> Indices:
        assert self.hop is not None
        # Greedy window build
        windows: Indices = []
        last = None
        for idx in self.stream_idx:
            if last is None or idx - last >= self.hop:
                if idx - self.time_dim + 1 < 0:
                    continue
                
                start = int(idx - self.time_dim + 1)
                end = int(idx)  # inclusive

                windows.append((start, end))
                last = idx

        return windows

    def _get_centered_window(self) -> Indices:
        '''Work only in PURE stable phoneme regions as specified by stream index'''
        length = self.neurogram.shape[0]
        
        valid = np.zeros(length, dtype=bool)
        valid[self.stream_idx.astype(int)] = True

        # detect run starts/ends (inclusive) within valid mask where phoneme stays constant
        prev_valid = np.roll(valid, 1); prev_valid[0] = False
        next_valid = np.roll(valid, -1); next_valid[-1] = False

        prev_same = prev_valid & (self.phoneme == np.roll(self.phoneme, 1))
        next_same = next_valid & (self.phoneme == np.roll(self.phoneme, -1))

        starts = np.flatnonzero(valid & ~prev_same)
        ends   = np.flatnonzero(valid & ~next_same)

        assert starts.size == ends.size, "Mismatch between run starts and ends!"
        # (rare) safety: realign if mismatch
        # if starts.size != ends.size:
        #     i = 0; s_list = []; e_list = []
        #     while i < length:
        #         while i < length and not valid[i]: i += 1
        #         if i >= length: break
        #         j = i + 1
        #         p = self.phoneme[i]
        #         while j < length and valid[j] and self.phoneme[j] == p: j += 1
        #         s_list.append(i); e_list.append(j - 1); i = j
        #     starts = np.asarray(s_list, dtype=int); ends = np.asarray(e_list, dtype=int)

        L = ends - starts + 1

        if self.hop is None:
            # keep only runs with length >= self.time_dim
            long_mask = L >= self.time_dim
            if not long_mask.any():
                return []

            s = starts[long_mask]
            e = ends[long_mask]
            c = (s + e) // 2

            st = np.maximum(s, c - self.time_dim // 2)
            st = np.minimum(st, e - self.time_dim + 1)
            st = np.maximum(st, s)
            en = st + self.time_dim - 1

            return list(map(tuple, np.stack([st, en], axis=1).astype(int)))

        # hop is int: sliding windows
        long_mask = L >= self.time_dim
        if not long_mask.any():
            return []

        n_windows = 1 + (L[long_mask] - self.time_dim) // self.hop
        total = int(n_windows.sum())

        run_ids = np.repeat(np.nonzero(long_mask)[0], n_windows)

        cum = np.cumsum(n_windows)
        block_starts = np.repeat(cum - n_windows, n_windows)
        pos_in_run = np.arange(total) - block_starts

        win_starts = starts[run_ids] + pos_in_run * self.hop
        win_ends   = win_starts + self.time_dim - 1
        
        return list(map(tuple, np.stack([win_starts, win_ends], axis=1).astype(int)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        start, end = self.indices[idx]
        x_win = self.neurogram[start:end+1] # [T, 150]
        p_win = self.phoneme[start:end+1] # phoneme index
        
        x = torch.from_numpy(x_win.T.copy()).unsqueeze(0) # (1, 150, T)
        label = torch.from_numpy(p_win.copy())
        return x, label
        

def make_loader(
    neurogram_path: Path | str,
    phoneme_path: Path | str,
    stream_path: Path | str,
    
    strategy: str  = 'centered',
    
    n_attrs: int = 40,
    val_split: float = 0.2,
    time_dim: int = 100,
    hop: int | None = None,

    batch_size: int = 60,
    shuffle: bool = True,
    num_workers: int = 0,
    
) -> Tuple[NeurogramDataset, Optional[DataLoader], Optional[DataLoader]]:
    neurogram_np = np.load( neurogram_path )
    phoneme_np = np.load( phoneme_path )
    stream_np = np.load( stream_path )
    
    ds = NeurogramDataset(
        neurogram_np, phoneme_np, stream_np,
        strategy=strategy,
        n_attrs=n_attrs, val_split=val_split,
        time_dim=time_dim, hop=hop
    )

    train_ds, val_ds = ds.train_val_split()

    if train_ds is not None:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=shuffle,
            drop_last=False, num_workers=num_workers
        )
    else: train_loader = None
    
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=shuffle,
            drop_last=False, num_workers=num_workers
        )
    else: val_loader = None

    return ds, train_loader, val_loader