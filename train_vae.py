import os
import json
import random
import argparse

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, MultiScaleStructuralSimilarityIndexMeasure


from utils import VAEDataset
from vae_model import VAE, VAELarge
from vae_loss import DualChannelPerceptualLoss, compute_vae_loss, group_independence_loss



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, default="/n/netscratch/kdbrantley_lab/Lab/jiajunh/autopet/data/for_vae/train")
    parser.add_argument("--test_data_path", type=str, default="/n/netscratch/kdbrantley_lab/Lab/jiajunh/autopet/data/for_vae/train")
    parser.add_argument("--log_path", type=str, default="/n/netscratch/kdbrantley_lab/Lab/jiajunh/autopet/ams_563_project/logs")
    parser.add_argument("--ckpt_dir", type=str, default="/n/netscratch/kdbrantley_lab/Lab/jiajunh/autopet/checkpoints")
    parser.add_argument("--data_augment", action="store_true")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--test_num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--test_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["cosine", "linear", "constant"])
    parser.add_argument("--warmup_frac", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--evaluate_freq", type=int, default=1)
    parser.add_argument("--test_chunk_size", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--kl_coef", type=float, default=1e-4)
    parser.add_argument("--kl_warmup_frac", type=float, default=0.25)

    parser.add_argument("--vae_choice", type=str, default="large", choices=["large", "plain"])
    parser.add_argument("--base_channels", type=int, default=8, choices=[4, 8, 16])
    parser.add_argument("--latent_dim", type=int, default=18, choices=[12, 18, 24], help="should be 3*k, k is int")
    parser.add_argument("--shared_channels", type=int, default=64, choices=[64, 128])

    parser.add_argument("--perception_loss_slices", type=int, default=16) # <= 32, otherwise oom, <=16 for large
    parser.add_argument("--perception_chunk_size", type=int, default=32)
    parser.add_argument("--perceptual_weight", type=float, default=0.1)
    parser.add_argument("--independent_weight", type=float, default=3.0)
    args = parser.parse_args()

    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def append_log(path, data):
    with open(path, "a") as f:
        f.write(json.dumps(data) + "\n")


def save_checkpoint(args, model, epoch):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt = {
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "args": vars(args),
    }
    path = os.path.join(args.ckpt_dir, f"epoch_{epoch+1:03d}.pt")
    torch.save(ckpt, path)
    print(f"Saved checkpoint: {path}")


def load_data(args):
    train_dataset = VAEDataset(
        data_dir=args.train_data_path,
        augment=args.data_augment,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_dataset = VAEDataset(
        data_dir=args.test_data_path,
        augment=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.test_num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def kl_divergence(mu, logvar):
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


@torch.no_grad()
def evaluate(args, model, data_loader):
    # PSNR, MS-SSIM
    model.eval()
    psnr_fn = PeakSignalNoiseRatio(data_range=2.0).to(args.device)
    ssim_fn = MultiScaleStructuralSimilarityIndexMeasure(data_range=2.0).to(args.device)

    for batch in tqdm(data_loader):
        batch = batch.to(args.device)
        B, C, D, H, W = batch.shape

        recon, mu, logvar = model(batch, sample_posterior=False)

        x_all = batch.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        r_all = recon.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        
        chunk_size = args.test_chunk_size
        for start in range(0, r_all.shape[0], chunk_size):
            end = min(start + chunk_size, r_all.shape[0])
            psnr_fn.update(r_all[start:end], x_all[start:end])
            ssim_fn.update(r_all[start:end], x_all[start:end])

    psnr = psnr_fn.compute().item()
    ssim = ssim_fn.compute().item()
    # print(f"PSNR={psnr:.4f}  MS-SSIM={ssim:.4f}")
    return psnr, ssim


def train(args, model, data_loader, optimizer, scheduler, perceptual_loss_fn, epoch):
    model.train()

    total_loss = 0.0
    total_vae_loss = 0.0
    total_kl_loss = 0.0
    total_gp_loss = 0.0

    kl_weight = get_kl_weight(args, epoch)
    sample_posterior = kl_weight > 0.0

    pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

    for batch in pbar:
        batch = batch.to(args.device, non_blocking=True)

        recon, mu, logvar = model(batch, sample_posterior=sample_posterior)
        vae_loss = compute_vae_loss(
            recon,
            batch,
            perceptual_loss_fn,
            perceptual_weight=args.perceptual_weight,
            n_slices=args.perception_loss_slices,
            chunk_size=args.perception_chunk_size,
        )
        gp_loss = group_independence_loss(mu)
        

        if kl_weight > 0.0:
            kl_loss = kl_divergence(mu, logvar)
        else:
            kl_loss = torch.zeros((), device=batch.device)

        loss = vae_loss + args.independent_weight * gp_loss  + kl_weight * kl_loss

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        bs = batch.size(0)
        total_loss += loss.item() * bs
        total_vae_loss += vae_loss.item() * bs
        total_kl_loss += kl_loss.item() * bs
        total_gp_loss += gp_loss.item() * bs

        pbar.set_postfix(
            total_loss=f"{loss.item():.3f}",
            vae_loss=f"{vae_loss.item():.3f}",
            kl_loss=f"{kl_loss.item():.3f}",
            gp_loss=f"{gp_loss.item():.3f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    updates = len(data_loader.dataset)
    avg_loss = total_loss / updates
    avg_vae_loss = total_vae_loss / updates
    avg_kl_loss = total_kl_loss / updates
    avg_gp_loss = total_gp_loss / updates

    return avg_loss, avg_vae_loss, avg_kl_loss, avg_gp_loss


if __name__ == "__main__":
    args = parse_args()
    print(args)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(args.log_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.log_path, "test"), exist_ok=True)

    file_name = f"vae_{args.vae_choice}_latent_{args.latent_dim}_bchannels_{args.base_channels}_schannels_{args.shared_channels}_pslices_{args.perception_loss_slices}.jsonl"
    train_log_path = os.path.join(args.log_path, "train", file_name)
    test_log_path = os.path.join(args.log_path, "test", file_name)

    if args.vae_choice == "plain":
        vae = VAE(
            in_channels=2,
            out_channels=2,
            base_channels=args.base_channels,
            shared_channels=args.shared_channels,
            latent_dim=args.latent_dim,
        ).to(device)

    else:
        vae = VAELarge(
            in_channels=2,
            out_channels=2,
            base_channels=args.base_channels,
            shared_channels=args.shared_channels,
            latent_dim=args.latent_dim,
            lk_3d_kernel=7,
            lk_2d_kernel=15
        ).to(device)

    train_loader, test_loader = load_data(args)

    optimizer = optim.AdamW(
        vae.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=1e-5,
    )

    total_steps = args.epochs * len(train_loader)
    scheduler = get_lr_scheduler(optimizer, args, total_steps)

    perceptual_loss_fn = DualChannelPerceptualLoss().to(device).eval()


    for epoch in range(args.epochs):
        print("-"*30, "Training", "-"*30)
        train_loss, train_vae, train_kl, train_group = train(
            args, vae, train_loader, optimizer, scheduler, perceptual_loss_fn, epoch
        )
        append_log(train_log_path, {
            "epoch": epoch + 1,
            "total_loss": train_loss,
            "vae_loss": train_vae,
            "kl_loss": train_kl,
            "group_loss": train_group,
        })

        save_checkpoint(args, vae, epoch)

        print("-"*30, "Testing", "-"*30)
        if epoch % args.evaluate_freq == 0:
            psnr, ssim = evaluate(args, vae, test_loader)

            append_log(test_log_path, {
                "epoch": epoch + 1,
                "psnr": psnr,
                "ssim": ssim, 
            })


