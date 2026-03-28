import os
import argparse
import pickle


import torch
import torch.nn.functional as F
import numpy as np
from datasets import Dataset
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source_path", type=str, default="/n/netscratch/kdbrantley_lab/Lab/jiajunh/autopet/data/processed/Dataset001_FDGPETCT_autopet3/nnUNetPlans_3d_fullres/")
    parser.add_argument("--target_path", type=str, default="/n/netscratch/kdbrantley_lab/Lab/jiajunh/autopet/data/for_vae/train/")
    parser.add_argument("--resample_size", type=int, default=256)

    args = parser.parse_args()
    return args


def resize_3d(img, target_size=(256, 256, 256)):
    x = torch.from_numpy(img).float().unsqueeze(0)
    x = F.interpolate(
        x,
        size=target_size,
        mode="trilinear",
        align_corners=False
    )
    return x.squeeze(0).numpy()


def load_one_date(file_name, resample_size):
    data = np.load(f"{file_name}.npz")
    img = data["data"]

    # print(f'data: {data["data"].shape}')
    # print(f'seg: {data["seg"].shape}')
    # print(np.mean(img[0]), np.max(img[0]), np.min(img[0]))
    # print(np.mean(img[1]), np.max(img[1]), np.min(img[1]))

    reshape_img = resize_3d(img, target_size=(resample_size, resample_size, resample_size))
    # print(torch.mean(reshape_img), torch.max(reshape_img), torch.min(reshape_img))
    return reshape_img


if __name__ == "__main__":
    args = parse_args()

    files = os.listdir(args.source_path)
    file_names = sorted(list(set([f.split(".")[0] for f in files])))
    # print(len(file_names))

    os.makedirs(args.target_path, exist_ok=True)

    for file_name in tqdm(file_names):
        file_path = os.path.join(args.source_path, file_name)
        img = load_one_date(file_path, args.resample_size)
        save_path = os.path.join(args.target_path, f"{file_name}.npz")

        np.savez_compressed(
            save_path,
            image=img.astype(np.float32)
        )
        # break

