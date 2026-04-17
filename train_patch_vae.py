import os
import json
import random
import argparse
import math

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, MultiScaleStructuralSimilarityIndexMeasure


from utils import VAEDataset, FullImagePatchDataset3D, PatchDataset3D
from patch_vae_model import PatchVAE3D, PatchDiscriminator3D
from vae_loss import DualChannelPerceptualLoss, compute_vae_loss, group_independence_loss



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, default="/n/netscratch/kdbrantley_lab/Lab/jiajunh/autopet/data/for_vae/train")
    parser.add_argument("--train_label_path", type=str, default=None)
    parser.add_argument("--test_data_path", type=str, default="/n/netscratch/kdbrantley_lab/Lab/jiajunh/autopet/data/for_vae/train")
    parser.add_argument("--test_label_path", type=str, default=None)
    parser.add_argument("--log_path", type=str, default="/n/netscratch/kdbrantley_lab/Lab/jiajunh/autopet/ams_563_project/logs")
    parser.add_argument("--ckpt_dir", type=str, default="/n/netscratch/kdbrantley_lab/Lab/jiajunh/autopet/checkpoints")
    parser.add_argument("--data_augment", action="store_true")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--test_num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--use_gan", action="store_true")

    parser.add_argument("--batch_size", type=int, default=1) # fix = 1
    parser.add_argument("--minibatch_size", type=int, default=64)
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
    parser.add_argument("--kl_coef", type=float, default=1e-5)
    parser.add_argument("--kl_warmup_frac", type=float, default=0.2)

    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--d_base_channels", type=int, default=16)
    parser.add_argument("--latent_dim", type=int, default=16)

    parser.add_argument("--perception_loss_slices", type=int, default=4)
    parser.add_argument("--perception_chunk_size", type=int, default=16)
    parser.add_argument("--perceptual_weight", type=float, default=1)
    parser.add_argument("--g_loss_weight", type=float, default=1)
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


def save_checkpoint(args, model, discriminator, epoch):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt = {
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "args": vars(args),
    }
    if discriminator is not None:
        ckpt["discriminator"] = discriminator.state_dict()

    path = os.path.join(args.ckpt_dir, f"patch_vae_epoch_{epoch+1:03d}.pt")
    torch.save(ckpt, path)
    print(f"Saved checkpoint: {path}")


def load_data(args):

    train_dataset = FullImagePatchDataset3D(
        image_dir=args.train_data_path, label_dir=args.train_label_path, 
        patch_size=args.patch_size, stride=args.stride,
    )

    # train_dataset = PatchDataset3D(
    #     image_dir=args.train_data_path, label_dir=args.train_label_path, 
    #     patch_size=args.patch_size, stride=args.stride,
    # )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_dataset = FullImagePatchDataset3D(
        image_dir=args.test_data_path, label_dir=args.test_label_path, 
        patch_size=args.patch_size, stride=args.stride,
    )

    # test_dataset = PatchDataset3D(
    #     image_dir=args.test_data_path, label_dir=args.test_label_path, 
    #     patch_size=args.patch_size, stride=args.stride,
    # )

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


def d_hinge_loss(real_logits, fake_logits):
    loss_real = F.relu(1.0 - real_logits).mean()
    loss_fake = F.relu(1.0 + fake_logits).mean()
    return loss_real + loss_fake

def g_hinge_loss(fake_logits):
    return -fake_logits.mean()


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
    model.eval()

    psnr_fn = PeakSignalNoiseRatio(data_range=2.0).to(args.device)

    ssim_fn = MultiScaleStructuralSimilarityIndexMeasure(
        data_range=2.0,
        betas=(0.3, 0.3, 0.4),
    ).to(args.device)

    for batch in tqdm(data_loader, desc="Evaluating"):
        patches = batch["patch"].to(args.device, non_blocking=True)   # (1, N, C, P, P, P)
        coords  = batch["coord"].to(args.device, non_blocking=True)
        starts  = batch["start"].to(args.device, non_blocking=True)

        B, N, C, P, _, _ = patches.shape
        assert B == 1

        patches = patches.view(N, C, P, P, P)
        coords  = coords.view(N, 3)
        starts  = starts.view(N, 3)

        # ---- forward in chunks ----
        chunk_size = args.test_chunk_size

        for start_idx in range(0, N, chunk_size):
            end = min(start_idx + chunk_size, N)

            p = patches[start_idx:end]     # (k, C, P, P, P)
            c = coords[start_idx:end]

            recon, _, _ = model(p, c, sample_posterior=False)

            # ---- compute metrics directly (skip GT stitching) ----
            x_all = p.permute(0, 2, 1, 3, 4).reshape(-1, C, P, P)
            r_all = recon.permute(0, 2, 1, 3, 4).reshape(-1, C, P, P)

            psnr_fn.update(r_all, x_all)
            ssim_fn.update(r_all, x_all)

    psnr = psnr_fn.compute().item()
    ssim = ssim_fn.compute().item()

    return psnr, ssim



def train(args, model, discriminator, data_loader, optimizer, optimizer_d,
          scheduler, scheduler_d, perceptual_loss_fn, epoch):
    model.train()
    total_loss = total_vae_loss = total_kl_loss = total_g_loss = total_d_loss = 0.0
    n_updates = 0
    kl_weight = get_kl_weight(args, epoch)
    sample_posterior = kl_weight > 0

    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

    for batch in pbar:
        patches = batch["patch"].to(args.device, non_blocking=True)
        coords  = batch["coord"].to(args.device, non_blocking=True)
        B, N, C, P, _, _ = patches.shape
        patches = patches.view(B * N, C, P, P, P)
        coords  = coords.view(B * N, 3)
        perm    = torch.randperm(patches.shape[0], device=patches.device)
        patches, coords = patches[perm], coords[perm]

        n_minibatches = math.ceil(patches.shape[0] / args.minibatch_size)
        g_loss = d_loss = torch.zeros((), device=patches.device)

        d_loss_accum = 0.0
        # ---- discriminator accumulation pass ----
        if args.use_gan:
            optimizer_d.zero_grad(set_to_none=True)
            for i in range(0, patches.shape[0], args.minibatch_size):
                p = patches[i:i + args.minibatch_size]
                c = coords[i:i + args.minibatch_size]
                with torch.no_grad():
                    recon_d, _, _ = model(p, c, sample_posterior=sample_posterior)
                d_loss = d_hinge_loss(discriminator(p), discriminator(recon_d))
                (d_loss / n_minibatches).backward()
                d_loss_accum += d_loss.item()

            nn.utils.clip_grad_norm_(discriminator.parameters(), args.grad_clip)
            optimizer_d.step()
            scheduler_d.step()

        # ---- VAE accumulation pass ----
        optimizer.zero_grad(set_to_none=True)
        vae_loss_accum = kl_loss_accum = g_loss_accum = 0.0

        for i in range(0, patches.shape[0], args.minibatch_size):
            p = patches[i:i + args.minibatch_size]
            c = coords[i:i + args.minibatch_size]

            recon, mu, logvar = model(p, c, sample_posterior=sample_posterior)
            vae_loss = compute_vae_loss(
                recon, p, perceptual_loss_fn,
                perceptual_weight=args.perceptual_weight,
                n_slices=args.perception_loss_slices,
                chunk_size=args.perception_chunk_size,
            )
            kl_loss = kl_divergence(mu, logvar) if kl_weight > 0 else torch.zeros((), device=p.device)
            loss = vae_loss + kl_weight * kl_loss

            if args.use_gan:
                g_loss = g_hinge_loss(discriminator(recon))
                loss = loss + args.g_loss_weight * g_loss

            if not torch.isfinite(loss):
                print(f"Non-finite loss at minibatch {i}, skipping")
                continue

            n_total = patches.shape[0]
            n_this = p.shape[0]
            (loss * n_this / n_total).backward()

            vae_loss_accum += vae_loss.item()
            kl_loss_accum += kl_loss.item()
            g_loss_accum += g_loss.item() if args.use_gan else 0.0

        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        # average over minibatches for logging
        d_loss_avg = d_loss_accum / n_minibatches
        vae_loss_avg = vae_loss_accum / n_minibatches
        kl_loss_avg = kl_loss_accum / n_minibatches
        g_loss_avg = g_loss_accum / n_minibatches
        total_loss_avg = vae_loss_avg + kl_weight * kl_loss_avg + args.g_loss_weight * g_loss_avg

        n_updates += 1
        total_loss += total_loss_avg
        total_vae_loss += vae_loss_avg
        total_kl_loss += kl_loss_avg
        total_g_loss += g_loss_avg
        total_d_loss += d_loss_avg if args.use_gan else 0.0

        pbar.set_postfix(
            vae=f"{vae_loss_avg:.3f}",
            kl=f"{kl_loss_avg:.3f}",
            g=f"{g_loss_avg:.3f}",
            d=f"{d_loss_avg:.3f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )
        # break

    n = max(n_updates, 1)
    return {
        "total_loss": total_loss / n,
        "vae_loss": total_vae_loss / n,
        "kl_loss": total_kl_loss / n,
        "g_loss": total_g_loss / n,
        "d_loss": total_d_loss / n,
    }


if __name__ == "__main__":
    args = parse_args()
    print(args)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(args.log_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.log_path, "test"), exist_ok=True)

    file_name = f"patch_vae_latent_{args.latent_dim}_bchannels_{args.base_channels}.jsonl"
    train_log_path = os.path.join(args.log_path, "train", file_name)
    test_log_path = os.path.join(args.log_path, "test", file_name)

    vae = PatchVAE3D(
        in_channels=2,
        base_channels=args.base_channels,
        latent_dim=args.latent_dim,
    ).to(device)

    # print(vae)

    train_loader, test_loader = load_data(args)

    optimizer = optim.AdamW(
        vae.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=1e-5,
    )


    total_steps = args.epochs * len(train_loader)
    scheduler = get_lr_scheduler(optimizer, args, total_steps)


    discriminator = None
    optimizer_d = None
    scheduler_d = None

    if args.use_gan:
        discriminator = PatchDiscriminator3D(
            in_channels=2,
            base_channels=args.d_base_channels,
        ).to(device)

        optimizer_d = optim.AdamW(
            discriminator.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            eps=1e-5,
        )

        scheduler_d = get_lr_scheduler(optimizer_d, args, total_steps)

    perceptual_loss_fn = DualChannelPerceptualLoss().to(device).eval()

    for epoch in range(args.epochs):
        print("-"*30, "Training", "-"*30)
        losses = train(
            args, vae, discriminator, train_loader, optimizer, optimizer_d, scheduler, scheduler_d, perceptual_loss_fn, epoch
        )

        append_log(train_log_path, {
            "epoch": epoch + 1,
            "total_loss": losses["total_loss"],
            "vae_loss": losses["vae_loss"],
            "kl_loss": losses["kl_loss"],
            "g_loss": losses["g_loss"],
            "d_loss": losses["d_loss"],
        })

        save_checkpoint(args, vae, discriminator, epoch)

        print("-"*30, "Testing", "-"*30)
        if epoch % args.evaluate_freq == 0:
            psnr, ssim = evaluate(args, vae, test_loader)

            append_log(test_log_path, {
                "epoch": epoch + 1,
                "psnr": psnr,
                "ssim": ssim, 
            })


