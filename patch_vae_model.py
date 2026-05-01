import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=min(groups, in_channels), num_channels=in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=min(groups, out_channels), num_channels=out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)

        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)

        return x + residual


class Downsample3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # stride-2 conv halves spatial size
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=3, stride=2, padding=1
        )

    def forward(self, x):
        return self.conv(x)


class Upsample3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        x = self.conv(x)
        return x



class PositionMLP(nn.Module):
    def __init__(self, latent_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.to_gamma = nn.Linear(hidden_dim, latent_dim)
        self.to_beta = nn.Linear(hidden_dim, latent_dim)

    def forward(self, coords):
        h = self.net(coords)
        gamma = self.to_gamma(h)
        beta = self.to_beta(h)
        return gamma, beta



class PatchVAE3D(nn.Module):
    def __init__(
        self,
        in_channels=2,
        base_channels=16,
        latent_dim=16,
    ):
        super().__init__()

        self.enc_in = ResBlock3D(in_channels, base_channels)
        self.enc_down1 = Downsample3D(base_channels, base_channels * 2)
        self.enc_res1 = ResBlock3D(base_channels * 2, base_channels * 2)

        self.enc_down2 = Downsample3D(base_channels * 2, base_channels * 4)
        self.enc_res2 = ResBlock3D(base_channels * 4, base_channels * 4)

        self.to_mu = nn.Conv3d(base_channels * 4, latent_dim, kernel_size=1)
        self.to_logvar = nn.Conv3d(base_channels * 4, latent_dim, kernel_size=1)

        # Position conditioning
        self.pos_mlp = PositionMLP(latent_dim=latent_dim, hidden_dim=64)

        # Decoder
        self.dec_in = ResBlock3D(latent_dim, base_channels * 4)
        self.dec_up1 = Upsample3D(base_channels * 4, base_channels * 2)
        self.dec_res1 = ResBlock3D(base_channels * 2, base_channels * 2)

        self.dec_up2 = Upsample3D(base_channels * 2, base_channels)
        self.dec_res2 = ResBlock3D(base_channels, base_channels)

        self.out_conv = nn.Conv3d(base_channels, in_channels, kernel_size=1)

    def encode(self, x):
        h = self.enc_in(x)
        h = self.enc_down1(h)
        h = self.enc_res1(h)
        h = self.enc_down2(h)
        h = self.enc_res2(h)

        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def apply_position_conditioning(self, z, coords):
        gamma, beta = self.pos_mlp(coords)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        z = z * (1.0 + gamma) + beta
        return z

    def decode(self, z):
        h = self.dec_in(z)
        h = self.dec_up1(h)
        h = self.dec_res1(h)
        h = self.dec_up2(h)
        h = self.dec_res2(h)
        recon = self.out_conv(h)
        return recon

    def get_latent(self, x, coords, sample_posterior=False):
        mu, logvar = self.encode(x)
        if sample_posterior:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        z = self.apply_position_conditioning(z, coords)
        return z

    def forward(self, x, coords, sample_posterior=False):
        mu, logvar = self.encode(x)
        if sample_posterior:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        z = self.apply_position_conditioning(z, coords)
        recon = self.decode(z)
        return recon, mu, logvar

    

class PatchSegmentation3D(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=1,
        base_channels=16,
        latent_dim=16,
    ):
        super().__init__()

        self.enc_in = ResBlock3D(in_channels, base_channels)
        self.enc_down1 = Downsample3D(base_channels, base_channels * 2)
        self.enc_res1 = ResBlock3D(base_channels * 2, base_channels * 2)

        self.enc_down2 = Downsample3D(base_channels * 2, base_channels * 4)
        self.enc_res2 = ResBlock3D(base_channels * 4, base_channels * 4)

        self.to_latent = nn.Conv3d(base_channels * 4, latent_dim, kernel_size=1)

        # Same position conditioning as VAE
        self.pos_mlp = PositionMLP(latent_dim=latent_dim, hidden_dim=64)

        # Decoder: latent -> segmentation logits
        self.dec_in = ResBlock3D(latent_dim, base_channels * 4)

        self.dec_up1 = Upsample3D(base_channels * 4, base_channels * 2)
        self.dec_res1 = ResBlock3D(base_channels * 2, base_channels * 2)

        self.dec_up2 = Upsample3D(base_channels * 2, base_channels)
        self.dec_res2 = ResBlock3D(base_channels, base_channels)
        self.out_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def encode(self, x):
        h = self.enc_in(x)
        h = self.enc_down1(h)
        h = self.enc_res1(h)
        h = self.enc_down2(h)
        h = self.enc_res2(h)
        z = self.to_latent(h)
        return z


    def apply_position_conditioning(self, z, coords):
        gamma, beta = self.pos_mlp(coords)
        gamma = gamma[:, :, None, None, None]
        beta = beta[:, :, None, None, None]
        return z * (1.0 + gamma) + beta

    def decode(self, z):
        h = self.dec_in(z)
        h = self.dec_up1(h)
        h = self.dec_res1(h)
        h = self.dec_up2(h)
        h = self.dec_res2(h)
        logits = self.out_conv(h)
        return logits

    def forward(self, x, coords):
        z = self.encode(x)
        z = self.apply_position_conditioning(z, coords)
        logits = self.decode(z)
        return logits



class PatchDiscriminator3D(nn.Module):
    def __init__(self, in_channels=2, base_channels=16):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),   # 64 -> 32
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),  # 32 -> 16
            nn.GroupNorm(8, base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),  # 16 -> 8
            nn.GroupNorm(8, base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),  # 8 -> 4
            nn.GroupNorm(8, base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(base_channels * 8, 1, kernel_size=4, stride=1, padding=0),  # 4 -> 1
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(x.shape[0], -1)

    

class PatchSegDiscriminator3D(nn.Module):
    def __init__(self, image_channels=2, mask_channels=1, base_channels=16):
        super().__init__()

        in_channels = image_channels + mask_channels

        self.net = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),  # 64 -> 32
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),  # 32 -> 16
            nn.GroupNorm(8, base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),  # 16 -> 8
            nn.GroupNorm(8, base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),  # 8 -> 4
            nn.GroupNorm(8, base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(base_channels * 8, 1, kernel_size=4, stride=1, padding=0),  # 4 -> 1
        )

    def forward(self, image, mask):
        x = torch.cat([image, mask], dim=1)
        out = self.net(x)
        return out.view(image.shape[0], -1)



class PatchSegmentation3DWithGlobal(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=1,
        base_channels=16,
        latent_dim=16,
    ):
        super().__init__()

        self.enc_in = ResBlock3D(in_channels, base_channels)
        self.enc_down1 = Downsample3D(base_channels, base_channels * 2)
        self.enc_res1 = ResBlock3D(base_channels * 2, base_channels * 2)

        self.enc_down2 = Downsample3D(base_channels * 2, base_channels * 4)
        self.enc_res2 = ResBlock3D(base_channels * 4, base_channels * 4)

        self.to_latent = nn.Conv3d(base_channels * 4, latent_dim, kernel_size=1)

        # Same position conditioning as VAE
        self.pos_mlp = PositionMLP(latent_dim=latent_dim, hidden_dim=64)

        # Decoder: latent -> segmentation logits
        self.dec_in = ResBlock3D(latent_dim, base_channels * 4)

        self.dec_up1 = Upsample3D(base_channels * 4, base_channels * 2)
        self.dec_res1 = ResBlock3D(base_channels * 2, base_channels * 2)

        self.dec_up2 = Upsample3D(base_channels * 2, base_channels)
        self.dec_res2 = ResBlock3D(base_channels, base_channels)
        self.out_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)


        self.coord_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )

        self.global_encoder = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d(1),
        )

        self.global_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 4, latent_dim * 2),
        )

    def encode(self, x):
        h = self.enc_in(x)
        h = self.enc_down1(h)
        h = self.enc_res1(h)
        h = self.enc_down2(h)
        h = self.enc_res2(h)
        z = self.to_latent(h)
        return z


    def apply_position_conditioning(self, z, coords):
        gamma, beta = self.pos_mlp(coords)
        gamma = gamma[:, :, None, None, None]
        beta = beta[:, :, None, None, None]
        return z * (1.0 + gamma) + beta

    def decode(self, z):
        h = self.dec_in(z)
        h = self.dec_up1(h)
        h = self.dec_res1(h)
        h = self.dec_up2(h)
        h = self.dec_res2(h)
        logits = self.out_conv(h)
        return logits

    def predict_coord(self, z):
        z_pool = z.mean(dim=(2, 3, 4))
        return self.coord_head(z_pool)
        
    def get_conditioned_latent(self, x, coords, global_x=None):
        z = self.encode(x)
        z = self.apply_position_conditioning(z, coords)

        if global_x is not None:
            gb = self.global_encoder(global_x)
            gamma_beta = self.global_mlp(gb)
            gamma, beta = gamma_beta.chunk(2, dim=1)
            gamma = gamma[:, :, None, None, None]
            beta = beta[:, :, None, None, None]
            z = z * (1.0 + gamma) + beta
        return z

    def forward(self, x, coords, global_x=None):
        z = self.encode(x)
        z = self.apply_position_conditioning(z, coords)

        if global_x is not None:
            gb = self.global_encoder(global_x)
            gamma_beta = self.global_mlp(gb)
            gamma, beta = gamma_beta.chunk(2, dim=1)
            gamma = gamma[:, :, None, None, None]
            beta = beta[:, :, None, None, None]
            z = z * (1.0 + gamma) + beta

        logits = self.decode(z)
        return logits

