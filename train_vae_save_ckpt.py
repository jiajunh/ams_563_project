import os
import random
import argparse

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

import csv
import json
import matplotlib.pyplot as plt
import torch.optim as optim


from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import VAEDataset
from vae_model import VAE



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/PETCT/dataset/for_vae/train_preprocessed")
    parser.add_argument("--do_normalize", action="store_true")
    parser.add_argument("--data_augment", action="store_true")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)


    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["cosine", "linear", "constant"])
    parser.add_argument("--warmup_frac", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--kl_coef", type=float, default=1e-4)
    parser.add_argument("--kl_warmup_frac", type=float, default=0.25)
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse"])


    parser.add_argument("--base_channels", type=int, default=4, choices=[4, 8, 16])
    parser.add_argument("--latent_hw", type=int, default=128, choices=[128, 64])
    parser.add_argument("--first_kernel_size", type=int, default=5, choices=[3, 5])
    parser.add_argument("--latent_dim", type=int, default=24, choices=[24, 30, 36, 48], help="should be 3*k, k is int")
    parser.add_argument("--shared_channels", type=int, default=64, choices=[128, 64])
    args = parser.parse_args()

    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_data(args):
    dataset = VAEDataset(
        data_dir=args.data_path,
        normalize=args.do_normalize,
        augment=args.data_augment
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return loader


def kl_divergence(mu, logvar):
    # mean over batch
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl.mean()


def compute_reconstruction_loss(loss_type, x, y):
    if loss_type == "mse":
        return F.mse_loss(x, y)
    return F.l1_loss(x, y)


def get_lr_scheduler(optimizer, args, total_steps):
    warmup_steps = int(args.warmup_frac * total_steps)
    min_lr = args.lr * args.min_lr_ratio
    if args.lr_scheduler == "cosine":
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return cosine * (1 - args.min_lr_ratio) + args.min_lr_ratio
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_kl_weight(args, epoch):
    warmup_epochs = int(args.kl_warmup_frac * args.epochs)
    if warmup_epochs <= 0:
        return args.kl_coef
    if epoch < warmup_epochs:
        return 0.0
    anneal_epochs = max(1, args.epochs - warmup_epochs)
    progress = (epoch - warmup_epochs + 1) / anneal_epochs
    progress = min(1.0, progress)
    return args.kl_coef * progress


def train(args, model, data_loader, optimizer, scheduler, epoch):
    model.train()

    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0

    kl_weight = get_kl_weight(args, epoch)
    sample_posterior = kl_weight > 0.0

    pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

    for batch in pbar:
        batch = batch.to(args.device, non_blocking=True)

        print("batch shape:", batch.shape)

        recon, mu, logvar = model(batch, sample_posterior=sample_posterior)
        recon_loss = compute_reconstruction_loss(args.loss_type, recon, batch)

        if kl_weight > 0.0:
            kl_loss = kl_divergence(mu, logvar)
        else:
            kl_loss = torch.zeros((), device=batch.device)

        loss = recon_loss + kl_weight * kl_loss

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        bs = batch.size(0)
        total_loss += loss.item() * bs
        total_recon_loss += recon_loss.item() * bs
        total_kl_loss += kl_loss.item() * bs

        pbar.set_postfix(
            total_loss=f"{loss.item():.4f}",
            recon_loss=f"{recon_loss.item():.4f}",
            kl_loss=f"{kl_loss.item():.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    updates = len(data_loader.dataset)
    avg_loss = total_loss / updates
    avg_recon_loss = total_recon_loss / updates
    avg_kl_loss = total_kl_loss / updates

    print(
        f"Epoch: {epoch + 1}: "
        f"loss: {avg_loss:.6f}, "
        f"recon: {avg_recon_loss:.6f}, "
        f"kl: {avg_kl_loss:.6f}"
    )

    return avg_loss, avg_recon_loss, avg_kl_loss


if __name__ == "__main__":
    args = parse_args()
    print(args)

    set_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    args.device = device
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    vae = VAE(
        in_channels=2,
        out_channels=2,
        base_channels=args.base_channels,
        shared_channels=args.shared_channels,
        latent_dim=args.latent_dim,
        latent_hw=args.latent_hw,
        first_kernel_size=args.first_kernel_size,
    ).to(device)

    data_loader = load_data(args)

    optimizer = optim.AdamW(
        vae.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=1e-5,
    )

    total_steps = args.epochs * len(data_loader)
    scheduler = get_lr_scheduler(optimizer, args, total_steps)

    # Save paths
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    best_model_path = os.path.join(save_dir, "best_model.pth")
    final_model_path = os.path.join(save_dir, "final_model.pth")

    best_loss = float("inf")

    for epoch in range(args.epochs):
        train_loss, train_recon, train_kl = train(
            args, vae, data_loader, optimizer, scheduler, epoch
        )

        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": vae.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_loss,
                "args": vars(args),
            }, best_model_path)
            print(f"Best model saved at epoch {epoch + 1}, loss={best_loss:.6f}")

    # Save final model
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": vae.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_loss": best_loss,
        "args": vars(args),
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")


# if __name__ == "__main__":
#     args = parse_args()
#     print(args)

#     set_seed(args.seed)
#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # args.device = device

#     if torch.cuda.is_available():
#         device = torch.device("cuda:0")
#     else:
#         device = torch.device("cpu")

#     args.device = device
#     print("Using device:", device)
#     if device.type == "cuda":
#         print("GPU:", torch.cuda.get_device_name(0))


#     vae = VAE(
#         in_channels=2,
#         out_channels=2,
#         base_channels=args.base_channels,
#         shared_channels=args.shared_channels,
#         latent_dim=args.latent_dim,
#         latent_hw=args.latent_hw,
#         first_kernel_size=args.first_kernel_size,
#     ).to(device)
#     # print(vae)
#     data_loader = load_data(args)

#     optimizer = optim.AdamW(
#         vae.parameters(),
#         lr=args.lr,
#         weight_decay=args.weight_decay,
#         eps=1e-5,
#     )

#     total_steps = args.epochs * len(data_loader)
#     scheduler = get_lr_scheduler(optimizer, args, total_steps)


#     for epoch in range(args.epochs):
#         train_loss, train_recon, train_kl = train(
#             args, vae, data_loader, optimizer, scheduler, epoch
#         )

