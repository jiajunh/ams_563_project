import os
import json
import random
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from transformers import AutoModel

from utils import SegmentDataset
from vae_model import VAE, VAELarge, SegDecoder3D, Decoder, DecoderLarge


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_image_path", type=str, default="/n/netscratch/kdbrantley_lab/Lab/jiajunh/autopet/data/for_vae/train")
    parser.add_argument("--train_label_path", type=str, default="/n/netscratch/kdbrantley_lab/Lab/jiajunh/autopet/data/for_vae/train")
    parser.add_argument("--test_image_path", type=str, default="/n/netscratch/kdbrantley_lab/Lab/jiajunh/autopet/data/for_vae/train")
    parser.add_argument("--test_label_path", type=str, default="/n/netscratch/kdbrantley_lab/Lab/jiajunh/autopet/data/for_vae/train")
    parser.add_argument("--log_path", type=str, default="/n/netscratch/kdbrantley_lab/Lab/jiajunh/autopet/ams_563_project/logs")
    parser.add_argument("--vae_ckpt_path", type=str, default="/n/netscratch/kdbrantley_lab/Lab/jiajunh/autopet/checkpoints")
    parser.add_argument("--ckpt_dir", type=str, default="/n/netscratch/kdbrantley_lab/Lab/jiajunh/autopet/checkpoints/decoder")
    parser.add_argument("--use_dino", action="store_true")
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
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--dino_multi_scale", type=int, default=3)
    parser.add_argument("--focal_gamma", type=float, default=2)
    parser.add_argument("--focal_alpha", type=float, default=0.95)
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--focal_weight", type=float, default=50.0)
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


def load_vae(args):
    ckpt = torch.load(args.vae_ckpt_path)
    # print(ckpt["args"])
    if ckpt["args"]["vae_choice"] == "plain":
        vae = VAE(
            in_channels=2,
            out_channels=2,
            base_channels=ckpt["args"]["base_channels"],
            shared_channels=ckpt["args"]["shared_channels"],
            latent_dim=ckpt["args"]["latent_dim"],
        )
    else:
        vae = VAELarge(
            in_channels=2,
            out_channels=2,
            base_channels=ckpt["args"]["base_channels"],
            shared_channels=ckpt["args"]["shared_channels"],
            latent_dim=ckpt["args"]["latent_dim"],
            lk_3d_kernel=7,
            lk_2d_kernel=15
        )
    vae.load_state_dict(ckpt["model"])
    vae.requires_grad_(False)
    vae.to(device)
    vae.eval()
    return vae, ckpt["args"]


def load_data(args):
    train_dataset = SegmentDataset(
        image_dir=args.train_image_path,
        label_dir=args.train_label_path,
        augment=args.data_augment,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )


    test_dataset = SegmentDataset(
        image_dir=args.test_image_path,
        label_dir=args.test_label_path,
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


def load_dino(args):
    dino_name = "facebook/dinov2-base"

    dino = AutoModel.from_pretrained(dino_name)
    dino.requires_grad_(False)
    dino.to(args.device)
    dino.eval()
    return dino


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


class MultiScaleDINO(nn.Module):
    def __init__(self, dino, k, 
        layer_indices=[3, 7, 11], proj_dim=128, out_dim=128):
        super().__init__()
        self.dino = dino
        self.layer_indices = layer_indices
        self.k = k

        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(768, proj_dim, 1),
                nn.BatchNorm2d(proj_dim),
                nn.ReLU()
            ) for _ in layer_indices
        ])

        self.slice_compress = nn.Sequential(
            nn.Conv2d(k * proj_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )
    
    def forward(self, x, target_size):
        Bk = x.shape[0]
        B = Bk // self.k

        with torch.no_grad():
            out = self.dino(pixel_values=x, output_hidden_states=True)

        feats = []
        for i, layer_idx in enumerate(self.layer_indices):
            h = out.hidden_states[layer_idx][:, 1:, :]
            h = h.transpose(1, 2).reshape(Bk, 768, 16, 16)
            h = F.interpolate(h, size=target_size, mode="bilinear", align_corners=False)
            h = self.projections[i](h)
            feats.append(h)

        scale_fused = torch.stack(feats, dim=0).sum(0)
        _, C, H, W = scale_fused.shape
        slice_fused = scale_fused.reshape(B, self.k * C, H, W)
        return self.slice_compress(slice_fused)  



def dice_loss(pred_sig, target, smooth=1e-5):
    intersection = (pred_sig * target).sum(dim=(2, 3, 4))
    dice = 1 - (2 * intersection + smooth) / (
        pred_sig.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4)) + smooth
    )
    return dice.mean()


def focal_loss(pred, target, gamma=2.0, alpha=0.8):
    bce = F.binary_cross_entropy_with_logits(pred, target.float(), reduction="none")
    prob = torch.sigmoid(pred)
    p_t = prob * target + (1 - prob) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    focal = alpha_t * (1 - p_t) ** gamma * bce
    return focal.mean()


def seg_loss(pred, target, dice_w=1.0, focal_w=1.0, focal_gamma=2.0, focal_alpha=0.8):
    pred_sig = torch.sigmoid(pred)
    loss_dice  = dice_w  * dice_loss(pred_sig, target)
    loss_focal = focal_w * focal_loss(pred, target, gamma=focal_gamma, alpha=focal_alpha)

    return loss_dice + loss_focal, {
        "dice":  loss_dice.item(),
        "focal": loss_focal.item(),
    }


def save_checkpoint(args, epoch, seg_decoder, ms_dino=None):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    if ms_dino is not None:
        ckpt = {
            "epoch": epoch + 1,
            "seg_decoder": seg_decoder.state_dict(),
            "ms_dino": ms_dino.state_dict(),
            "args": vars(args),
        }
    else:
        ckpt = {
            "epoch": epoch + 1,
            "seg_decoder": seg_decoder.state_dict(),
            "args": vars(args),
        }

    path = os.path.join(args.ckpt_dir, f"epoch_{epoch+1:03d}.pt")
    torch.save(ckpt, path)
    print(f"Saved checkpoint: {path}")


def train(args, vae_encoder, seg_decoder, data_loader, optimizer, scheduler, epoch, trainable_params, ms_dino=None):
    seg_decoder.train()
    if ms_dino is not None:
        ms_dino.dino.eval()
        ms_dino.projections.train()
        ms_dino.slice_compress.train()

    total_loss = 0.0
    total_dice_loss = 0.0
    total_focal_loss = 0.0

    pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
    for batch_imgs, batch_labels in pbar:
        batch = batch_imgs.to(args.device, non_blocking=True)
        y = batch_labels.to(args.device)

        # b, dim, 224, 224
        with torch.no_grad():
            mu, _ = vae_encoder(batch)
            mu = mu.detach()
        
        if ms_dino is not None:
            B, L, H, W = mu.shape
            k = L // 3
            x = mu.view(B, k, 3, H, W).reshape(B * k, 3, H, W)
            dino_feature = ms_dino(x, target_size=(H, W))
            # print(dino_feature.shape)
            pred = seg_decoder(mu, dino_feature)

        else:
            pred = seg_decoder(mu)

        loss, metrics = seg_loss(
            pred, y,
            dice_w=args.dice_weight, 
            focal_w=args.focal_weight, 
            focal_gamma=args.focal_gamma, 
            focal_alpha=args.focal_alpha,
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_dice_loss += metrics["dice"]
        total_focal_loss += metrics["focal"]

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            dice=f"{metrics['dice']:.4f}",
            focal=f"{metrics['focal']:.4f}",
        )

    updates = len(data_loader)
    avg_loss = total_loss / updates
    avg_dice_loss = total_dice_loss / updates
    avg_focal_loss = total_focal_loss / updates

    return avg_loss, avg_dice_loss, avg_focal_loss


@torch.no_grad()
def evaluate(args, data_loader, vae_encoder, seg_decoder, ms_dino=None):
    vae_encoder.eval()
    seg_decoder.eval()
    if ms_dino is not None:
        ms_dino.eval()

    total_dice, total_samples = 0.0, 0
    total_loss = 0.0

    pbar = tqdm(data_loader, desc="Evaluating")
    for batch_imgs, batch_labels in pbar:
        batch = batch_imgs.to(args.device, non_blocking=True)
        masks = batch_labels.to(args.device, non_blocking=True)

        mu, _ = vae_encoder(batch)
        B, L, H, W = mu.shape

        if ms_dino is not None:
            k = L // 3
            x = mu.view(B, k, 3, H, W).reshape(B * k, 3, H, W)
            dino_feat = ms_dino(x, target_size=(H, W))
            pred = seg_decoder(mu, dino_feat)
        else:
            pred = seg_decoder(mu)

        pred_bin = (torch.sigmoid(pred) > 0.5).float()
        intersection = (pred_bin * masks).sum(dim=(2, 3, 4))
        dice = (2 * intersection + 1e-5) / (
            pred_bin.sum(dim=(2, 3, 4)) + masks.sum(dim=(2, 3, 4)) + 1e-5
        )
        batch_dice = dice.mean().item()
        total_dice    += batch_dice * B
        total_samples += B

        pbar.set_postfix(dice=f"{batch_dice:.4f}")

    return total_dice / total_samples


if __name__ == "__main__":
    args = parse_args()
    print(args)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(args.log_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.log_path, "test"), exist_ok=True)

    file_name = f"dino_{args.use_dino}_dino_multi_scale_{args.dino_multi_scale}.jsonl"
    train_log_path = os.path.join(args.log_path, "train", file_name)
    test_log_path = os.path.join(args.log_path, "test", file_name)
    
    vae, vae_args = load_vae(args)
    latent_dim = vae_args["latent_dim"]
    vae_choice = vae_args["vae_choice"]
    vae_base_channels = vae_args["base_channels"]
    shared_channels = vae_args["shared_channels"]

    train_loader, test_loader = load_data(args)
    
    ms_dino = None
    if args.use_dino:
        dino = load_dino(args)
        # print(dino)
        K = latent_dim // 3

        layer_indices = [7, 11]
        if args.dino_multi_scale == 3:
            layer_indices = [3, 7, 11]

        ms_dino = MultiScaleDINO(
            dino,
            k=K,
            layer_indices=layer_indices,
            proj_dim=128,
            out_dim=128
        ).to(device)

        seg_decoder = SegDecoder3D(
            num_classes=1,
            latent_dim=latent_dim,
            dino_out_dim=128,
            base_channels=4,
        ).to(device)

        trainable_params = (
            list(ms_dino.projections.parameters()) +
            list(ms_dino.slice_compress.parameters()) +
            list(seg_decoder.parameters())
        )

    else:
        if vae_choice == "plain":
            seg_decoder = Decoder(
                out_channels=1,
                base_channels=vae_base_channels,
                shared_channels=shared_channels,
                latent_dim=latent_dim,
            ).to(device)
        else:
            seg_decoder = DecoderLarge(
                out_channels=1,
                base_channels=vae_base_channels,
                shared_channels=shared_channels,
                latent_dim=latent_dim,
            ).to(device)

        vae_dec_state = vae.decoder.state_dict()
        seg_dec_state = seg_decoder.state_dict()
        for k in seg_dec_state:
            if k in vae_dec_state and seg_dec_state[k].shape == vae_dec_state[k].shape:
                seg_dec_state[k] = vae_dec_state[k]
        seg_decoder.load_state_dict(seg_dec_state)

        trainable_params = (
            list(seg_decoder.parameters())
        )
    

    optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=1e-5,
    )

    total_steps = args.epochs * len(train_loader)
    scheduler = get_lr_scheduler(optimizer, args, total_steps)

    for epoch in range(args.epochs):
        print("-"*30, "Training", "-"*30)
        loss, train_dice_loss, train_focal_loss = train(
            args, vae.encoder, seg_decoder, 
            train_loader, optimizer, scheduler, epoch, 
            trainable_params, ms_dino=ms_dino) 

        append_log(train_log_path, {
            "epoch": epoch + 1,
            "total_loss": loss,
            "dice_loss": train_dice_loss,
            "focal_loss": train_focal_loss,
        })

        save_checkpoint(args, epoch, seg_decoder, ms_dino=ms_dino)

        print("-"*30, "Testing", "-"*30)
        if (epoch + 1) % args.evaluate_freq == 0:
            dice = evaluate(args, test_loader, vae.encoder, seg_decoder, ms_dino=ms_dino)

            append_log(test_log_path, {
                "epoch": epoch + 1,
                "dice": dice,
            })






