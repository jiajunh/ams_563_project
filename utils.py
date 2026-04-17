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
            "patch": patches,
            "coord": coords,
            "start": starts,
        }

        if label_patches is not None:
            out["label_patches"] = torch.stack(label_patches, dim=0)

        return out



class PatchDataset3D(Dataset):
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

        self.index_map = []

        target_D, target_H, target_W = 224, 224, 224
        d_starts = self._get_starts(target_D)
        h_starts = self._get_starts(target_H)
        w_starts = self._get_starts(target_W)

        for vol_idx in range(len(self.files)):
            for sd in d_starts:
                for sh in h_starts:
                    for sw in w_starts:
                        self.index_map.append((vol_idx, sd, sh, sw))

    def __len__(self):
        return len(self.index_map)

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
        return torch.tensor([
            (sd + half) / D,
            (sh + half) / H,
            (sw + half) / W,
        ], dtype=torch.float32)

    def _resize(self, x, y=None):
        target_shape = (224,224,224)

        if tuple(x.shape[1:]) != target_shape:
            x = x.unsqueeze(0) 
            x = F.interpolate(
                x,
                size=target_shape,
                mode="trilinear",
                align_corners=False
            )
            x = x.squeeze(0)

            if y is not None:
                y = y.unsqueeze(0).unsqueeze(0)
                y = F.interpolate(
                    y,
                    size=target_shape,
                    mode="nearest"
                )
                y = y.squeeze(0).squeeze(0)

        return x, y

    def __getitem__(self, idx):
        vol_idx, sd, sh, sw = self.index_map[idx]
        file_name = self.files[vol_idx]

        img_path = os.path.join(self.image_dir, file_name)
        label_path = None
        if self.label_dir is not None:
            label_path = os.path.join(self.label_dir, file_name)

        img, label = self._load_npz(img_path, label_path)

        x = torch.from_numpy(img).to(self.dtype)
        y = torch.from_numpy(label).to(self.dtype) if label is not None else None

        x, y = self._resize(x, y)

        C, D, H, W = x.shape
        P = self.patch_size

        patch = x[:, sd:sd+P, sh:sh+P, sw:sw+P]
        coord = self._compute_coord(sd, sh, sw, D, H, W)
        start = torch.tensor([sd, sh, sw], dtype=torch.long)

        out = {
            "patch": patch,
            "coord": coord,
            "start": start,
            "volume_idx": vol_idx,
        }

        if y is not None:
            label_patch = y[sd:sd+P, sh:sh+P, sw:sw+P].unsqueeze(0)  # [1, P, P, P]
            out["label"] = label_patch

        return out

