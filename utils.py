import os
import numpy as np
import torch
from torch.utils.data import Dataset


class VAEDataset(Dataset):
    def __init__(self, data_dir, normalize=False, augment=False, eps=1e-6, dtype=torch.float32):
        self.data_dir = data_dir
        self.normalize = normalize
        self.augment = augment
        self.eps = eps
        self.dtype = dtype

        self.files = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npz")
        ])

    def __len__(self):
        return len(self.files)

    def _load_npz(self, path):
        data = np.load(path)
        x = data["image"]
        return x

    def __getitem__(self, idx):
        file_path = self.files[idx]
        x = self._load_npz(file_path)
        x = torch.from_numpy(x).to(self.dtype)
        return x