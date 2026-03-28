import torch
import torch.nn as nn
import torch.nn.functional as F



def make_group_norm(num_channels, max_groups=8):
    for g in [8, 4, 2, 1]:
        if g <= max_groups and num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


class ResBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = make_group_norm(in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = make_group_norm(out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)



class DownsampleResBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride=(2, 2, 2)):
        super().__init__()
        self.norm1 = make_group_norm(in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )

        self.norm2 = make_group_norm(out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

        self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


class ResBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = make_group_norm(in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = make_group_norm(out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)



class UpResBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale_factor=(2, 2, 2)):
        super().__init__()
        self.scale_factor = scale_factor
        self.block = ResBlock3D(in_channels, out_channels)

    def forward(self, x):
        x = F.interpolate(
            x, scale_factor=self.scale_factor, mode="trilinear", align_corners=False
        )
        x = self.block(x)
        return x



class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=2,
        base_channels=8,
        shared_channels=128,
        latent_dim=36,
        latent_hw=128,
        first_kernel_size=5,
    ):
        super().__init__()
        self.latent_hw = latent_hw
        self.base_channels = base_channels
        self.shared_channels = shared_channels
        self.latent_dim = latent_dim

        self.conv_in = nn.Conv3d(
            in_channels, base_channels, kernel_size=first_kernel_size,
            stride=1, padding=first_kernel_size // 2
        )

        # 256^3 -> 128 x 256 x 256
        self.down1 = DownsampleResBlock3D(
            base_channels, base_channels, stride=(2, 1, 1)
        )

        # 128 x 256 x 256 -> 64 x 256 x 256
        self.down2 = DownsampleResBlock3D(
            base_channels, base_channels * 2, stride=(2, 1, 1)
        )

        # 64 x 256 x 256 -> 32 x 128 x 128
        self.down3 = DownsampleResBlock3D(
            base_channels * 2, base_channels * 4, stride=(2, 2, 2)
        )

        if latent_hw == 64:
            # 32 x 128 x 128 -> 16 x 64 x 64
            self.down4 = DownsampleResBlock3D(
                base_channels * 4, base_channels * 4, stride=(2, 2, 2)
            )
            depth_after_3d = 16
        else:
            self.down4 = None
            depth_after_3d = 32

        hidden_channels = base_channels * 4
        self.hidden_channels = hidden_channels
        self.depth_after_3d = depth_after_3d

        collapsed_channels = hidden_channels * depth_after_3d

        self.to_2d = nn.Sequential(
            nn.Conv2d(collapsed_channels, shared_channels, kernel_size=1),
            ResBlock2D(shared_channels, shared_channels),
        )
        self.quant_conv = nn.Conv2d(shared_channels, 2 * latent_dim, kernel_size=1)


    def forward(self, x):
        x = self.conv_in(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        if self.down4 is not None:
            x = self.down4(x)

        b, c, d, h, w = x.shape
        x = x.reshape(b, c * d, h, w)
        x = self.to_2d(x)

        moments = self.quant_conv(x)
        mu, logvar = torch.chunk(moments, 2, dim=1)
        return mu, logvar



class Decoder(nn.Module):
    def __init__(
        self,
        out_channels=2,
        base_channels=8,
        shared_channels=128,
        latent_dim=36,
        latent_hw=128,
    ):
        super().__init__()
        self.latent_hw = latent_hw
        self.base_channels = base_channels
        self.shared_channels = shared_channels
        self.latent_dim = latent_dim

        hidden_channels = base_channels * 4
        self.hidden_channels = hidden_channels

        if latent_hw == 64:
            depth_after_3d = 16
        else:
            depth_after_3d = 32
        self.depth_after_3d = depth_after_3d

        collapsed_channels = hidden_channels * depth_after_3d

        self.post_quant_conv = nn.Conv2d(latent_dim, shared_channels, kernel_size=1)

        self.from_2d = nn.Sequential(
            ResBlock2D(shared_channels, shared_channels),
            nn.Conv2d(shared_channels, collapsed_channels, kernel_size=1),
        )

        if latent_hw == 64:
            self.up1 = UpResBlock3D(hidden_channels, hidden_channels, scale_factor=(2, 2, 2))
            self.up2 = UpResBlock3D(hidden_channels, base_channels * 2, scale_factor=(2, 2, 2))
            self.up3 = UpResBlock3D(base_channels * 2, base_channels, scale_factor=(2, 1, 1))
            self.up4 = UpResBlock3D(base_channels, base_channels, scale_factor=(2, 1, 1))
        else:
            self.up1 = UpResBlock3D(hidden_channels, base_channels * 2, scale_factor=(2, 2, 2))
            self.up2 = UpResBlock3D(base_channels * 2, base_channels, scale_factor=(2, 1, 1))
            self.up3 = UpResBlock3D(base_channels, base_channels, scale_factor=(2, 1, 1))
            self.up4 = None

        self.norm_out = make_group_norm(base_channels)
        self.act_out = nn.SiLU()
        self.out_conv = nn.Conv3d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, z):
        x = self.post_quant_conv(z)
        x = self.from_2d(x)

        b, cd, h, w = x.shape
        x = x.view(b, self.hidden_channels, self.depth_after_3d, h, w)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        if self.up4 is not None:
            x = self.up4(x)

        x = self.out_conv(self.act_out(self.norm_out(x)))
        return x


class VAE(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=2,
        base_channels=8,
        shared_channels=128,
        latent_dim=36,
        latent_hw=128,
        first_kernel_size=5,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels=in_channels,
            base_channels=base_channels,
            shared_channels=shared_channels,
            latent_dim=latent_dim,
            latent_hw=latent_hw,
            first_kernel_size=first_kernel_size,
        )
        self.decoder = Decoder(
            out_channels=out_channels,
            base_channels=base_channels,
            shared_channels=shared_channels,
            latent_dim=latent_dim,
            latent_hw=latent_hw,
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, sample_posterior=True):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar) if sample_posterior else mu
        recon = self.decoder(z)
        return recon, mu, logvar
