import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGGFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        f = vgg.features
        self.slice1 = nn.Sequential(*f[:4])     # relu1_2 → 64ch
        self.slice2 = nn.Sequential(*f[4:9])    # relu2_2 → 128ch
        self.slice3 = nn.Sequential(*f[9:16])   # relu3_3 → 256ch
        # self.slice4 = nn.Sequential(*f[16:23])  # relu4_3 → 512ch
        # self.slice5 = nn.Sequential(*f[23:30])  # relu5_3 → 512ch
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        # h4 = self.slice4(h3)
        # h5 = self.slice5(h4)

        return h1, h2, h3
        # return h1, h2, h3, h4, h5


class DualChannelPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = VGGFeatures()
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def preprocess(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = (x + 1) / 2
        return (x - self.mean) / self.std

    def l2_normalize(self, x, eps=1e-8):
        return x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + eps)

    def channel_loss(self, recon_ch, real_ch):
        r = self.preprocess(recon_ch)
        r_feats = self.vgg(r)
        with torch.no_grad():
            x = self.preprocess(real_ch)
            x_feats = self.vgg(x)
        loss = 0.0
        for rf, xf in zip(r_feats, x_feats):
            loss = loss + (self.l2_normalize(rf) - self.l2_normalize(xf)).pow(2).sum(dim=1).mean()
        return loss

    def forward(self, recon, real):
        ct_loss  = self.channel_loss(recon[:, 0:1], real[:, 0:1])
        pet_loss = self.channel_loss(recon[:, 1:2], real[:, 1:2])
        return ct_loss + pet_loss


def compute_vae_loss(
    recon,
    real,
    perceptual_loss_fn,
    perceptual_weight=1.0,
    n_slices=None,
    chunk_size=1,
):
    B, C, D, H, W = real.shape
    rec_loss = torch.abs(recon - real).mean()

    if n_slices is None or n_slices >= D:
        indices = torch.arange(D, device=real.device)
    else:
        indices = torch.randperm(D, device=real.device)[:n_slices]

    recon_all = recon[:, :, indices].permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
    real_all = real[:, :, indices].permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)

    p_loss_sum = 0.0
    num_chunks = 0
    for start in range(0, recon_all.shape[0], chunk_size):
        end = min(start + chunk_size, recon_all.shape[0])
        p_loss_sum = p_loss_sum + perceptual_loss_fn(
            recon_all[start:end],
            real_all[start:end],
        )
        num_chunks += 1

    p_loss = p_loss_sum / num_chunks
    # print(f"rec_loss: {rec_loss.item():.4f}, p_loss: {p_loss.item():.4f}")
    return rec_loss + perceptual_weight * p_loss


def group_independence_loss(z, group_size=3, eps=1e-8):
    B, C, H, W = z.shape
    assert C % group_size == 0

    G = C // group_size
    z = z.view(B, G, group_size, H, W)
    z = z.reshape(B, G, -1)

    z = z - z.mean(dim=-1, keepdim=True)
    z = z / (z.norm(dim=-1, keepdim=True) + eps)

    sim = torch.matmul(z, z.transpose(1, 2))  # [B, G, G]

    eye = torch.eye(G, device=z.device).unsqueeze(0)
    off_diag = sim * (1 - eye)

    return (off_diag ** 2).mean()