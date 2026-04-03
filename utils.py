import os
import torch

import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset


class VAEDataset(Dataset):
    def __init__(self, data_dir, augment=False, dtype=torch.float32):
        self.data_dir = data_dir
        self.augment = augment
        self.dtype = dtype

        self.files = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npz")
        ])

        self.axis_perms = [
            (0, 1, 2, 3),
            (0, 2, 3, 1),
            (0, 3, 1, 2),
        ]

    def __len__(self):
        if self.augment:
            return len(self.files) * 3
        return len(self.files)

    def _load_npz(self, path):
        data = np.load(path)
        x = data["image"]
        return x

    def __getitem__(self, idx):
        if self.augment:
            file_idx = idx // 3
            perm_idx = idx % 3
        else:
            file_idx = idx
            perm_idx = 0

        file_path = self.files[file_idx]
        x = self._load_npz(file_path)
        x = torch.from_numpy(x).to(self.dtype)

        if x.shape[1:] != (224, 224, 224):
            x = x.unsqueeze(0)
            x = F.interpolate(
                x,
                size=(224, 224, 224),
                mode="trilinear",
                align_corners=False
            )
            x = x.squeeze(0)

        x = x.permute(*self.axis_perms[perm_idx]).contiguous()
        return x

