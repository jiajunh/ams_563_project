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

from utils import VAEDataset, FullImagePatchDataset3D, BalancedLesionPatchDataset3D
from patch_vae_model import PatchVAE3D, PatchDiscriminator3D, PatchSegmentation3D, PatchSegDiscriminator3D


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
    parser.add_argument("--dataset_type", type=str, default="all", choices=["all", "balanced"])
    parser.add_argument("--sample_patches_per_volume", type=int, default=64)
    parser.add_argument("--sample_positive_ratio", type=float, default=0.5)
    parser.add_argument("--sample_pos_jitter", type=int, default=20)

    parser.add_argument("--batch_size", type=int, default=1) # fix = 1
    parser.add_argument("--minibatch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=1)
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
    parser.add_argument("--kl_warmup_frac", type=float, default=0.2)

    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--d_base_channels", type=int, default=16)
    parser.add_argument("--latent_dim", type=int, default=16)


    parser.add_argument("--g_loss_weight", type=float, default=1)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--focal_alpha", type=float, default=0.95)
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--focal_weight", type=float, default=10.0)

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

    path = os.path.join(args.ckpt_dir, f"patch_segmentation_epoch_{epoch+1:03d}.pt")
    torch.save(ckpt, path)
    print(f"Saved checkpoint: {path}")


def load_data(args):

    if args.dataset_type == "balanced":
        train_dataset = BalancedLesionPatchDataset3D(
            image_dir=args.train_data_path,
            label_dir=args.train_label_path,
            patch_size=args.patch_size,
            patches_per_volume=args.sample_patches_per_volume,
            positive_ratio=args.sample_positive_ratio,
            pos_jitter=args.sample_pos_jitter,
        )
    else:
        train_dataset = FullImagePatchDataset3D(
            image_dir=args.train_data_path,
            label_dir=args.train_label_path,
            patch_size=args.patch_size,
            stride=args.stride,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_dataset = FullImagePatchDataset3D(
        image_dir=args.test_data_path,
        label_dir=args.test_label_path,
        patch_size=args.patch_size,
        stride=args.stride,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.test_num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


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


def dice_loss(logits, target, eps=1e-6):
    pred = torch.sigmoid(logits)

    dims = (1, 2, 3, 4)
    intersection = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims)

    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def focal_loss(logits, target, alpha=0.95, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)

    pt = prob * target + (1.0 - prob) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)

    loss = alpha_t * (1.0 - pt).pow(gamma) * bce
    return loss.mean()


def compute_seg_loss(logits, target, args):
    d_loss = dice_loss(logits, target)
    f_loss = focal_loss(
        logits,
        target,
        alpha=args.focal_alpha,
        gamma=args.focal_gamma,
    )

    loss = args.dice_weight * d_loss + args.focal_weight * f_loss
    return loss, d_loss, f_loss



@torch.no_grad()
def evaluate1(args, model, data_loader):
    model.eval()

    chunk_size = args.test_chunk_size
    P = args.patch_size

    total_dice = 0.0
    total_fpv = 0.0
    total_fnv = 0.0
    total_fp_ratio = 0.0
    total_fn_ratio = 0.0
    n_cases = 0

    for batch in tqdm(data_loader, desc="Evaluating"):
        x_full = batch["image"].to(args.device, non_blocking=True)
        y_full = batch["label"].to(args.device, non_blocking=True).float()
        patches = batch["patch"].to(args.device, non_blocking=True)
        coords = batch["coord"].to(args.device, non_blocking=True)
        starts = batch["start"].to(args.device, non_blocking=True)

        B, N, C, _, _, _ = patches.shape
        assert B == 1, "Please set --test_batch_size 1 for evaluation."

        x_full = x_full[0]
        y_full = y_full[0]

        if y_full.ndim == 3:
            y_full = y_full.unsqueeze(0)

        patches = patches[0]
        coords = coords[0]
        starts = starts[0]

        _, D, H, W = x_full.shape

        logit_full = torch.zeros((1, D, H, W), device=args.device)
        weight_full = torch.zeros((1, D, H, W), device=args.device)

        for start_idx in range(0, N, chunk_size):
            end_idx = min(start_idx + chunk_size, N)

            p = patches[start_idx:end_idx]
            c = coords[start_idx:end_idx]

            logits = model(p, c)

            patch_starts = starts[start_idx:end_idx]

            for j in range(logits.shape[0]):
                sd, sh, sw = patch_starts[j].tolist()

                logit_full[:, sd:sd+P, sh:sh+P, sw:sw+P] += logits[j]
                weight_full[:, sd:sd+P, sh:sh+P, sw:sw+P] += 1.0

        logit_full = logit_full / weight_full.clamp_min(1.0)

        pred = (torch.sigmoid(logit_full) > 0.5).float()
        target = (y_full > 0.5).float()

        inter = (pred * target).sum()
        pred_sum = pred.sum()
        target_sum = target.sum()

        fp = ((pred == 1) & (target == 0)).sum().float()
        fn = ((pred == 0) & (target == 1)).sum().float()

        voxel_volume = 1.0
        fpv = fp.float() * voxel_volume
        fnv = fn.float() * voxel_volume

        total_voxels = pred.numel()
        target_sum = target.sum().float()
        fp_ratio = fp / (total_voxels + 1e-6)
        fn_ratio = fn / (total_voxels + 1e-6) 

        dice = (2.0 * inter + 1e-6) / (pred_sum + target_sum + 1e-6)

        total_dice += dice.item()
        total_fp_ratio += fp_ratio.item()
        total_fn_ratio += fn_ratio.item()
        total_fpv += fpv.item()
        total_fpn += fpn.item()
        n_cases += 1

    n = max(n_cases, 1)

    return {
        "dice": total_dice / n,
        "fp": total_fp_ratio / n,
        "fn": total_fn_ratio / n,
        "fpv": total_fpv / n,
        "fnv": total_fpn / n,
    }


def evaluate(args, model, data_loader):
    model.eval()

    chunk_size = args.test_chunk_size
    P = args.patch_size

    total_dice = 0.0
    total_fpv = 0.0
    total_fnv = 0.0
    total_fp_ratio = 0.0
    total_fn_ratio = 0.0
    n_cases = 0

    for batch in tqdm(data_loader, desc="Evaluating"):
        x_full = batch["image"].to(args.device, non_blocking=True)
        y_full = batch["label"].to(args.device, non_blocking=True).float()
        patches = batch["patch"].to(args.device, non_blocking=True)
        coords = batch["coord"].to(args.device, non_blocking=True)
        starts = batch["start"].to(args.device, non_blocking=True)

        B, N, C, _, _, _ = patches.shape
        assert B == 1, "Please set --test_batch_size 1 for evaluation."

        x_full = x_full[0]
        y_full = y_full[0]

        if y_full.ndim == 3:
            y_full = y_full.unsqueeze(0)

        patches = patches[0]
        coords = coords[0]
        starts = starts[0]

        _, D, H, W = x_full.shape

        logit_full = torch.zeros((1, D, H, W), device=args.device)
        weight_full = torch.zeros((1, D, H, W), device=args.device)

        for start_idx in range(0, N, chunk_size):
            end_idx = min(start_idx + chunk_size, N)

            p = patches[start_idx:end_idx]
            c = coords[start_idx:end_idx]

            logits = model(p, c)

            patch_starts = starts[start_idx:end_idx]

            for j in range(logits.shape[0]):
                sd, sh, sw = patch_starts[j].tolist()

                logit_full[:, sd:sd+P, sh:sh+P, sw:sw+P] += logits[j]
                weight_full[:, sd:sd+P, sh:sh+P, sw:sw+P] += 1.0

        logit_full = logit_full / weight_full.clamp_min(1.0)

        pred = (torch.sigmoid(logit_full) > 0.5).float()
        target = (y_full > 0.5).float()

        inter = (pred * target).sum()
        pred_sum = pred.sum()
        target_sum = target.sum()

        fp = ((pred == 1) & (target == 0)).sum().float()
        fn = ((pred == 0) & (target == 1)).sum().float()

        voxel_volume = 1.0
        fpv = fp.float() * voxel_volume
        fnv = fn.float() * voxel_volume

        total_voxels = pred.numel()
        target_sum = target.sum().float()
        fp_ratio = fp / (total_voxels + 1e-6)
        fn_ratio = fn / (total_voxels + 1e-6)

        dice = (2.0 * inter + 1e-6) / (pred_sum + target_sum + 1e-6)

        total_dice += dice.item()
        total_fp_ratio += fp_ratio.item()
        total_fn_ratio += fn_ratio.item()
        total_fpv += fpv.item()
        total_fnv += fnv.item()
        n_cases += 1

    n = max(n_cases, 1)

    return {
        "dice": total_dice / n,
        "fp": total_fp_ratio / n,
        "fn": total_fn_ratio / n,
        "fpv": total_fpv / n,
        "fnv": total_fnv / n,
    }


def train(args, model, discriminator, data_loader, optimizer, optimizer_d,
          scheduler, scheduler_d, epoch):
    
    model.train()
    if discriminator is not None:
        discriminator.train()

    total_loss = 0.0
    total_seg_loss = 0.0
    total_dice_loss = 0.0
    total_focal_loss = 0.0
    total_g_loss = 0.0
    total_d_loss = 0.0
    n_updates = 0


    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

    for batch in pbar:
        patches = batch["patch"].to(args.device, non_blocking=True)
        labels = batch["label_patches"].to(args.device, non_blocking=True)
        coords = batch["coord"].to(args.device, non_blocking=True)

        B, N, C, P, _, _ = patches.shape

        patches = patches.view(B * N, C, P, P, P)
        labels = labels.view(B * N, 1, P, P, P).float()
        coords = coords.view(B * N, 3)

        perm = torch.randperm(patches.shape[0], device=patches.device)
        patches = patches[perm]
        labels = labels[perm]
        coords = coords[perm]

        n_total = patches.shape[0]
        n_minibatches = math.ceil(n_total / args.minibatch_size)
        # print(n_total)

        d_loss_accum = 0.0

        # ---- discriminator update ----
        if args.use_gan:
            optimizer_d.zero_grad(set_to_none=True)

            for i in range(0, n_total, args.minibatch_size):
                p = patches[i:i + args.minibatch_size]
                y = labels[i:i + args.minibatch_size]
                c = coords[i:i + args.minibatch_size]

                with torch.no_grad():
                    logits = model(p, c)
                    fake_mask = torch.sigmoid(logits)

                real_logits = discriminator(p, y)
                fake_logits = discriminator(p, fake_mask.detach())

                d_loss = d_hinge_loss(real_logits, fake_logits)
                (d_loss * p.shape[0] / n_total).backward()

                d_loss_accum += d_loss.item()

            nn.utils.clip_grad_norm_(discriminator.parameters(), args.grad_clip)
            optimizer_d.step()
            scheduler_d.step()

        # ---- Segmentation update ----
        optimizer.zero_grad(set_to_none=True)

        seg_loss_accum = 0.0
        dice_loss_accum = 0.0
        focal_loss_accum = 0.0
        g_loss_accum = 0.0

        for i in range(0, n_total, args.minibatch_size):
            p = patches[i:i + args.minibatch_size]
            y = labels[i:i + args.minibatch_size]
            c = coords[i:i + args.minibatch_size]

            logits = model(p, c)

            seg_loss, d_loss_seg, f_loss_seg = compute_seg_loss(logits, y, args)
            loss = seg_loss

            g_loss = torch.zeros((), device=args.device)

            if args.use_gan:
                fake_mask = torch.sigmoid(logits)
                fake_logits = discriminator(p, fake_mask)
                g_loss = g_hinge_loss(fake_logits)
                loss = loss + args.g_loss_weight * g_loss

            if not torch.isfinite(loss):
                print(f"Non-finite loss at minibatch {i}, skipping")
                continue

            (loss * p.shape[0] / n_total).backward()

            seg_loss_accum += seg_loss.item()
            dice_loss_accum += d_loss_seg.item()
            focal_loss_accum += f_loss_seg.item()
            g_loss_accum += g_loss.item() if args.use_gan else 0.0

        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        # average over minibatches for logging
        seg_loss_avg = seg_loss_accum / n_minibatches
        dice_loss_avg = dice_loss_accum / n_minibatches
        focal_loss_avg = focal_loss_accum / n_minibatches
        g_loss_avg = g_loss_accum / n_minibatches
        d_loss_avg = d_loss_accum / n_minibatches if args.use_gan else 0.0

        total_loss_avg = seg_loss_avg + args.g_loss_weight * g_loss_avg

        total_loss += total_loss_avg
        total_seg_loss += seg_loss_avg
        total_dice_loss += dice_loss_avg
        total_focal_loss += focal_loss_avg
        total_g_loss += g_loss_avg
        total_d_loss += d_loss_avg
        n_updates += 1

        pbar.set_postfix(
            seg=f"{seg_loss_avg:.4f}",
            dice=f"{dice_loss_avg:.4f}",
            focal=f"{focal_loss_avg:.4f}",
            g=f"{g_loss_avg:.4f}",
            d=f"{d_loss_avg:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )
        # break

    n = max(n_updates, 1)
    return {
        "total_loss": total_loss / n,
        "seg_loss": total_seg_loss / n,
        "dice_loss": total_dice_loss / n,
        "focal_loss": total_focal_loss / n,
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

    file_name = f"patch_segmentation_latent_{args.latent_dim}_bchannels_{args.base_channels}.jsonl"
    train_log_path = os.path.join(args.log_path, "train", file_name)
    test_log_path = os.path.join(args.log_path, "test", file_name)

    model = PatchSegmentation3D(
        in_channels=2,
        out_channels=1,
        base_channels=args.base_channels,
        latent_dim=args.latent_dim,
    ).to(device)
    

    train_loader, test_loader = load_data(args)

    optimizer = optim.AdamW(
        model.parameters(),
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
        discriminator = PatchSegDiscriminator3D(
            image_channels=2, 
            mask_channels=1, 
            base_channels=args.d_base_channels,
        ).to(device)

        optimizer_d = optim.AdamW(
            discriminator.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            eps=1e-5,
        )

        scheduler_d = get_lr_scheduler(optimizer_d, args, total_steps)


    for epoch in range(args.epochs):
        print("-"*30, "Training", "-"*30)
        losses = train(
            args, model, discriminator, train_loader, optimizer, optimizer_d, scheduler, scheduler_d, epoch
        )

        append_log(train_log_path, {
            "epoch": epoch + 1,
            "total_loss": losses["total_loss"],
            "seg_loss": losses["seg_loss"],
            "dice_loss": losses["dice_loss"],
            "focal_loss": losses["focal_loss"],
            "g_loss": losses["g_loss"],
            "d_loss": losses["d_loss"],
        })

        save_checkpoint(args, model, discriminator, epoch)

        print("-"*30, "Testing", "-"*30)
        if (epoch + 1) % args.evaluate_freq == 0:
            metrics = evaluate(args, model, test_loader)

            append_log(test_log_path, {
                "epoch": epoch + 1,
                "dice": metrics["dice"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "fpv": metrics["fpv"],
                "fnv": metrics["fnv"],
            })
