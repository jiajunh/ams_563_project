import torch
import torch.nn as nn
import torch.nn.functional as F


def make_group_norm(num_channels, max_groups=32):
    for g in [32, 16, 8, 4, 2, 1]:
        if g <= max_groups and num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
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
    def __init__(self, in_channels, out_channels, stride=(2, 2, 2)):
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
    def __init__(self, in_channels, out_channels):
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
    def __init__(self, in_channels, out_channels, scale_factor=(2, 2, 2)):
        super().__init__()
        self.scale_factor = scale_factor
        self.block = ResBlock3D(in_channels, out_channels)

    def forward(self, x):
        x = F.interpolate(
            x, scale_factor=self.scale_factor, mode="trilinear", align_corners=False
        )
        return self.block(x)


class LargeKernelBlock3D(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.dw   = nn.Conv3d(
            channels, channels,
            kernel_size=(1, kernel_size, kernel_size),
            padding=(0, pad, pad),
            groups=channels, 
        )
        self.pw   = nn.Conv3d(channels, channels, kernel_size=1)
        self.norm = make_group_norm(channels)
        self.act  = nn.SiLU()
 
    def forward(self, x):
        return x + self.pw(self.act(self.norm(self.dw(x))))
 
 
class LargeKernelBlock2D(nn.Module):
    def __init__(self, channels, kernel_size=15, dilation=2):
        super().__init__()
        pad = (kernel_size // 2) * dilation
        self.dw   = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=pad,
            dilation=dilation,
            groups=channels,
        )
        self.pw   = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = make_group_norm(channels)
        self.act  = nn.SiLU()

    def forward(self, x):
        return x + self.pw(self.act(self.norm(self.dw(x))))
 


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=2,
        base_channels=8,
        shared_channels=64,
        latent_dim=12,
    ):
        super().__init__()
        self.base_channels = base_channels
        self.conv_in = nn.Conv3d(
            in_channels, base_channels, kernel_size=3, padding=1
        )
        # 224 x 224 x 224 -> 112 x 224 x 224
        self.down1 = DownsampleResBlock3D(base_channels, base_channels, stride=(2, 1, 1))
        # 112 x 224 x 224 ->  56 x 224 x 224
        self.down2 = DownsampleResBlock3D(base_channels, base_channels * 2, stride=(2, 1, 1))
        #  56 x 224 x 224 ->  28 x 224 x 224
        self.down3 = DownsampleResBlock3D(base_channels * 2, base_channels * 4, stride=(2, 1, 1))
        #  28 x 224 x 224 ->  14 x 224 x 224
        self.down4 = DownsampleResBlock3D(base_channels * 4, base_channels * 4, stride=(2, 1, 1))
        #  14 x 224 x 224 ->   7 x 224 x 224
        self.down5 = DownsampleResBlock3D(base_channels * 4, base_channels * 4, stride=(2, 1, 1))

        collapsed_channels = base_channels * 4 * 7
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
        x = self.down4(x)
        x = self.down5(x)

        b, c, d, h, w = x.shape
        x = x.reshape(b, c * d, h, w)
        x = self.to_2d(x)
        mu, logvar = torch.chunk(self.quant_conv(x), 2, dim=1)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels=2,
        base_channels=8,
        shared_channels=64,
        latent_dim=12,
    ):
        super().__init__()
        hidden_channels = base_channels * 4
        self.hidden_channels = hidden_channels
        self.depth_after_3d = 7

        self.post_quant_conv = nn.Conv2d(latent_dim, shared_channels, kernel_size=1)
        self.from_2d = nn.Sequential(
            ResBlock2D(shared_channels, shared_channels),
            nn.Conv2d(shared_channels, hidden_channels * 7, kernel_size=1),
        )

        self.up1 = UpResBlock3D(hidden_channels, base_channels * 4, scale_factor=(2, 1, 1))
        self.up2 = UpResBlock3D(base_channels * 4, base_channels * 4, scale_factor=(2, 1, 1))
        self.up3 = UpResBlock3D(base_channels * 4, base_channels * 2, scale_factor=(2, 1, 1))
        self.up4 = UpResBlock3D(base_channels * 2, base_channels, scale_factor=(2, 1, 1))
        self.up5 = UpResBlock3D(base_channels, base_channels, scale_factor=(2, 1, 1))

        self.norm_out = make_group_norm(base_channels)
        self.act_out  = nn.SiLU()
        self.out_conv = nn.Conv3d(base_channels, out_channels, kernel_size=3, padding=1)


    def forward(self, z):
        x = self.post_quant_conv(z)
        x = self.from_2d(x)
        b, cd, h, w = x.shape
        x = x.view(b, self.hidden_channels, self.depth_after_3d, h, w)
        x = self.up1(x)  # (B, 32,  14, 224, 224)
        x = self.up2(x)  # (B, 32,  28, 224, 224)
        x = self.up3(x)  # (B, 16,  56, 224, 224)
        x = self.up4(x)  # (B,  8, 112, 224, 224)
        x = self.up5(x)  # (B,  8, 224, 224, 224)
        return self.out_conv(self.act_out(self.norm_out(x)))


class EncoderLarge(nn.Module):
    def __init__(
        self,
        in_channels=2,
        base_channels=8,
        shared_channels=64,
        latent_dim=12,
        lk_3d_kernel=7,
        lk_2d_kernel=15,
    ):
        super().__init__()
        C = base_channels
        self.conv_in = nn.Conv3d(in_channels, C, kernel_size=3, padding=1)
 
        self.down1 = DownsampleResBlock3D(C, C, stride=(2, 1, 1))
        self.down2 = DownsampleResBlock3D(C, C*2, stride=(2, 1, 1))
        self.down3 = DownsampleResBlock3D(C*2, C*4, stride=(2, 1, 1))
        self.down4 = DownsampleResBlock3D(C*4, C*4, stride=(2, 1, 1))
        self.down5 = DownsampleResBlock3D(C*4, C*4, stride=(2, 1, 1))
 
        self.large_kernel_3d = LargeKernelBlock3D(C*4, kernel_size=lk_3d_kernel)
        collapsed_channels = C * 4 * 7
 
        self.to_2d = nn.Sequential(
            make_group_norm(collapsed_channels),
            nn.SiLU(),
            nn.Conv2d(collapsed_channels, shared_channels, kernel_size=1),
            ResBlock2D(shared_channels, shared_channels),
            LargeKernelBlock2D(shared_channels, kernel_size=lk_2d_kernel, dilation=2),
        )
 
        self.quant_conv = nn.Conv2d(shared_channels, 2 * latent_dim, kernel_size=1)
 
    def forward(self, x):
        x = self.conv_in(x) 
        x = self.down1(x)      # [B, C,   112, 224, 224]
        x = self.down2(x)      # [B, C*2,  56, 224, 224]
        x = self.down3(x)      # [B, C*4,  28, 224, 224]
        x = self.down4(x)      # [B, C*4,  14, 224, 224]
        x = self.down5(x)      # [B, C*4,   7, 224, 224]
 
        x = self.large_kernel_3d(x)  # [B, C*4, 7, 224, 224] — spatial context
 
        b, c, d, h, w = x.shape
        x = x.reshape(b, c * d, h, w)  # [B, C*4*7, 224, 224]
        x = self.to_2d(x)       # [B, shared_channels, 224, 224]
        mu, logvar = torch.chunk(self.quant_conv(x), 2, dim=1)
        return mu, logvar        # each [B, latent_dim, 224, 224]
 

class DecoderLarge(nn.Module):
    def __init__(
        self,
        out_channels=2,
        base_channels=8,
        shared_channels=64,
        latent_dim=12,
    ):
        super().__init__()
        C = base_channels
        hidden_channels = C * 4
        self.hidden_channels  = hidden_channels
        self.depth_after_3d   = 7
 
        self.post_quant_conv = nn.Conv2d(latent_dim, shared_channels, kernel_size=1)
 
        self.from_2d = nn.Sequential(
            ResBlock2D(shared_channels, shared_channels),
            nn.Conv2d(shared_channels, hidden_channels * 7, kernel_size=1),
        )
 
        self.up1 = UpResBlock3D(hidden_channels, C*4, scale_factor=(2, 1, 1))
        self.up2 = UpResBlock3D(C*4, C*4, scale_factor=(2, 1, 1))
        self.up3 = UpResBlock3D(C*4, C*2, scale_factor=(2, 1, 1))
        self.up4 = UpResBlock3D(C*2, C, scale_factor=(2, 1, 1))
        self.up5 = UpResBlock3D(C, C, scale_factor=(2, 1, 1))
 
        self.norm_out = make_group_norm(C)
        self.act_out  = nn.SiLU()
        self.out_conv = nn.Conv3d(C, out_channels, kernel_size=3, padding=1)
 
    def forward(self, z):
        x = self.post_quant_conv(z) 
        x = self.from_2d(x) 
        b, cd, h, w = x.shape
        x = x.view(b, self.hidden_channels, self.depth_after_3d, h, w)
        x = self.up1(x)   # [B, C*4, 14,  224, 224]
        x = self.up2(x)   # [B, C*4, 28,  224, 224]
        x = self.up3(x)   # [B, C*2, 56,  224, 224]
        x = self.up4(x)   # [B, C,   112, 224, 224]
        x = self.up5(x)   # [B, C,   224, 224, 224]
        return self.out_conv(self.act_out(self.norm_out(x)))
 


class VAE(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=2,
        base_channels=8,
        shared_channels=64,
        latent_dim=12,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels=in_channels,
            base_channels=base_channels,
            shared_channels=shared_channels,
            latent_dim=latent_dim,
        )
        self.decoder = Decoder(
            out_channels=out_channels,
            base_channels=base_channels,
            shared_channels=shared_channels,
            latent_dim=latent_dim,
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x, sample_posterior=True):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar) if sample_posterior else mu
        recon = self.decoder(z)
        return recon, mu, logvar


class VAELarge(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=2,
        base_channels=8,
        shared_channels=64,
        latent_dim=12,
        lk_3d_kernel=7,
        lk_2d_kernel=15,
    ):
        super().__init__()
        self.encoder = EncoderLarge(
            in_channels=in_channels,
            base_channels=base_channels,
            shared_channels=shared_channels,
            latent_dim=latent_dim,
            lk_3d_kernel=lk_3d_kernel,
            lk_2d_kernel=lk_2d_kernel,
        )
        self.decoder = DecoderLarge(
            out_channels=out_channels,
            base_channels=base_channels,
            shared_channels=shared_channels,
            latent_dim=latent_dim,
        )
 
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)
 
    def forward(self, x, sample_posterior=True):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar) if sample_posterior else mu
        recon = self.decoder(z)
        return recon, mu, logvar