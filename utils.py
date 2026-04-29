import os
import torch
import random

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



class SegmentDataset(Dataset):
    def __init__(self, image_dir, label_dir, augment=False, dtype=torch.float32):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.augment = augment
        self.dtype = dtype

        self.files = sorted([
            f for f in os.listdir(label_dir)
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

    def _load_npz(self, img_path, label_path):
        img_data = np.load(img_path)
        img = img_data["image"]
        label_data = np.load(label_path)
        label = label_data["label"]
        return img, label

    def __getitem__(self, idx):
        if self.augment:
            file_idx = idx // 3
            perm_idx = idx % 3
        else:
            file_idx = idx
            perm_idx = 0

        file_path = self.files[file_idx]

        img_path = os.path.join(self.image_dir, file_path)
        label_path = os.path.join(self.label_dir, file_path)

        img, label = self._load_npz(img_path, label_path)
        x = torch.from_numpy(img).to(self.dtype)
        y = torch.from_numpy(label).to(self.dtype)

        if x.shape[1:] != (224, 224, 224):
            x = x.unsqueeze(0)
            x = F.interpolate(
                x,
                size=(224, 224, 224),
                mode="trilinear",
                align_corners=False
            )
            x = x.squeeze(0)

            y = y.unsqueeze(0)
            y = F.interpolate(
                y,
                size=(224, 224, 224),
                mode="nearest",
            )
            y = y.squeeze(0)

        x = x.permute(*self.axis_perms[perm_idx]).contiguous()
        y = y.permute(*self.axis_perms[perm_idx]).contiguous()
        return x, y



class FullImagePatchDataset3D(Dataset):
    def __init__(self, image_dir, label_dir=None, patch_size=64, stride=32):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.patch_size = patch_size
        self.stride = stride
        self.dtype = torch.float32

        self.files = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(".npz")
        ])


    def __len__(self):
        return len(self.files)


    def _load_npz(self, img_path, label_path=None):
        img_data = np.load(img_path)
        img = img_data["image"]

        label = None
        if label_path is not None:
            label_data = np.load(label_path)
            label = label_data["label"]
        return img, label


    def _get_starts(self, size):
        P = self.patch_size
        S = self.stride

        starts = list(range(0, size - P + 1, S))
        if starts[-1] != size - P:
            starts.append(size - P)
        return starts

    def _compute_coord(self, sd, sh, sw, D, H, W):
        half = self.patch_size / 2.0
        return [
            (sd + half) / D,
            (sh + half) / H,
            (sw + half) / W,
        ]

    def __getitem__(self, idx):
        file_name = self.files[idx]

        img_path = os.path.join(self.image_dir, file_name)
        label_path = None
        if self.label_dir is not None:
            label_path = os.path.join(self.label_dir, file_name)

        img, label = self._load_npz(img_path, label_path)

        x = torch.from_numpy(img).to(self.dtype)
        y = torch.from_numpy(label).to(self.dtype) if label is not None else None

        if y is not None:
            if y.ndim == 4 and y.shape[0] == 1:
                y = y.squeeze(0)
            elif y.ndim != 3:
                raise ValueError(f"Unexpected label shape: {y.shape}")

        if x.shape[1:] != (224, 224, 224):
            x = x.unsqueeze(0)
            x = F.interpolate(
                x,
                size=(224, 224, 224),
                mode="trilinear",
                align_corners=False
            )
            x = x.squeeze(0)

            if y is not None:
                y = y.unsqueeze(0)
                y = F.interpolate(
                    y,
                    size=(224, 224, 224),
                    mode="nearest",
                )
                y = y.squeeze(0)


        C, D, H, W = x.shape
        P = self.patch_size

        d_starts = self._get_starts(D)
        h_starts = self._get_starts(H)
        w_starts = self._get_starts(W)

        patches = []
        coords = []
        starts = []
        label_patches = [] if y is not None else None

        for sd in d_starts:
            for sh in h_starts:
                for sw in w_starts:
                    patch = x[:, sd:sd+P, sh:sh+P, sw:sw+P]
                    patches.append(patch)

                    coords.append(self._compute_coord(sd, sh, sw, D, H, W))
                    starts.append((sd, sh, sw))

                    if y is not None:
                        label_patch = y[sd:sd+P, sh:sh+P, sw:sw+P].unsqueeze(0)
                        label_patches.append(label_patch)

        patches = torch.stack(patches, dim=0) 
        coords = torch.tensor(coords, dtype=torch.float32)
        starts = torch.tensor(starts, dtype=torch.long)

        out = {
            "image": x, 
            "patch": patches,
            "coord": coords,
            "start": starts,
        }

        if label_patches is not None:
            out["label_patches"] = torch.stack(label_patches, dim=0)
            out["label"] = y 

        return out



class BalancedLesionPatchDataset3D(Dataset):
    def __init__(
        self,
        image_dir,
        label_dir,
        patch_size=64,
        patches_per_volume=64,
        positive_ratio=0.4,
        target_shape=(224, 224, 224),
        pos_jitter=24,
        dtype=torch.float32,
    ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.positive_ratio = positive_ratio
        self.target_shape = target_shape
        self.pos_jitter = pos_jitter
        self.dtype = dtype

        self.files = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(".npz")
        ])

    def __len__(self):
        return len(self.files)

    def _load_npz(self, img_path, label_path):
        img = np.load(img_path)["image"]
        label = np.load(label_path)["label"]
        return img, label

    def _normalize(self, img, label):
        x = torch.from_numpy(img).to(self.dtype)
        y = torch.from_numpy(label).to(self.dtype)

        if x.ndim == 3:
            x = x.unsqueeze(0)

        if y.ndim == 3:
            y = y.unsqueeze(0)
        elif y.ndim == 4 and y.shape[0] == 1:
            pass
        else:
            raise ValueError(f"Unexpected label shape: {y.shape}")

        if tuple(x.shape[1:]) != self.target_shape:
            x = F.interpolate(
                x.unsqueeze(0),
                size=self.target_shape,
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)

        if tuple(y.shape[1:]) != self.target_shape:
            y = F.interpolate(
                y.unsqueeze(0),
                size=self.target_shape,
                mode="nearest",
            ).squeeze(0)

        y = (y > 0.5).float()
        return x, y

    def _random_start(self, D, H, W):
        P = self.patch_size
        sd = torch.randint(0, D - P + 1, (1,)).item()
        sh = torch.randint(0, H - P + 1, (1,)).item()
        sw = torch.randint(0, W - P + 1, (1,)).item()
        return sd, sh, sw

    def _positive_start(self, y, D, H, W):
        P = self.patch_size
        lesion_voxels = torch.nonzero(y[0] > 0.5, as_tuple=False)

        if lesion_voxels.numel() == 0:
            return self._random_start(D, H, W)

        idx = torch.randint(0, lesion_voxels.shape[0], (1,)).item()
        d, h, w = lesion_voxels[idx].tolist()

        # Put lesion somewhere inside or near the patch, not always centered
        min_sd = max(0, d - P + 1 - self.pos_jitter)
        max_sd = min(d + self.pos_jitter, D - P)

        min_sh = max(0, h - P + 1 - self.pos_jitter)
        max_sh = min(h + self.pos_jitter, H - P)

        min_sw = max(0, w - P + 1 - self.pos_jitter)
        max_sw = min(w + self.pos_jitter, W - P)

        sd = torch.randint(min_sd, max_sd + 1, (1,)).item()
        sh = torch.randint(min_sh, max_sh + 1, (1,)).item()
        sw = torch.randint(min_sw, max_sw + 1, (1,)).item()

        return sd, sh, sw

    def _compute_coord(self, sd, sh, sw, D, H, W):
        half = self.patch_size / 2.0
        return [
            (sd + half) / D,
            (sh + half) / H,
            (sw + half) / W,
        ]

    def __getitem__(self, idx):
        file_name = self.files[idx]

        img_path = os.path.join(self.image_dir, file_name)
        label_path = os.path.join(self.label_dir, file_name)

        img, label = self._load_npz(img_path, label_path)
        x, y = self._normalize(img, label)

        C, D, H, W = x.shape
        P = self.patch_size

        n_pos = int(self.patches_per_volume * self.positive_ratio)
        n_neg = self.patches_per_volume - n_pos

        starts = []

        for _ in range(n_pos):
            starts.append(self._positive_start(y, D, H, W))

        for _ in range(n_neg):
            starts.append(self._random_start(D, H, W))

        random.shuffle(starts)

        patches = []
        label_patches = []
        coords = []

        for sd, sh, sw in starts:
            patches.append(x[:, sd:sd+P, sh:sh+P, sw:sw+P])
            label_patches.append(y[:, sd:sd+P, sh:sh+P, sw:sw+P])
            coords.append(self._compute_coord(sd, sh, sw, D, H, W))

        return {
            "image": x,
            "label": y,
            "patch": torch.stack(patches, dim=0),
            "label_patches": torch.stack(label_patches, dim=0),
            "coord": torch.tensor(coords, dtype=torch.float32),
            "start": torch.tensor(starts, dtype=torch.long),
        }
