import os
import numpy as np
import torch
from torch.utils.data import Dataset


class NeurogramDataset(Dataset):
    def __init__(self, data_dir, label_path, target_dim=(150, 100)):
        self.data_dir = data_dir
        self.target_dim = target_dim
        print("loading labels...")

        with open(label_path, "r") as f:
            _ = f.readline()
            self.label_name = f.readline().strip().split(" ")
            self.label_name = self.label_name[1:]
            self.label_size = len(self.label_name)
            self.labels = []
        
        self.data_files = [os.path.join(data_dir, f"{idx+1:06}.npy") for idx, _ in self.labels]


    def __len__(self):

        return len(self.labels)

    def __getitem__(self, index):
        data, mask = self._load_and_process_data(self.data_files[abs(index)], index < 0)
        label = self.labels[index][1]

        return data, label, mask

 

    def _load_and_process_data(self, file_path, flip=False):
        data = np.load(file_path)
        if data.max() != data.min():
            data = (data - data.min()) / (data.max() - data.min())  # Min-max scaling to [0, 1]

        if data.shape[0] != self.target_dim[0]:
            raise ValueError(f"Data shape mismatch: expected first dimension {self.target_dim[0]}, got {data.shape[0]}")
        original_length = data.shape[1]

        if original_length < self.target_dim[1]:
            padding = (0, self.target_dim[1] - original_length)
            data = np.pad(data, ((0, 0), padding), mode='constant')

        elif original_length > self.target_dim[1]:
            data = data[:, :self.target_dim[1]]

        mask = np.zeros((self.target_dim[0], self.target_dim[1]), dtype=np.float32)
        mask[:, :original_length] = 1.0

        if flip:
            data = np.flip(data, axis=1)
            mask = np.flip(mask, axis=1)

        data = torch.tensor(data, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        return data, mask