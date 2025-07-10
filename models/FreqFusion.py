import torch
import torch.nn as nn
import torch.nn.functional as F


class FreqFusion(nn.Module):
    def __init__(self,
                 hr_channels,
                 lr_channels,
                 compressed_channels=64,
                 feature_resample_group=4,
                 kernel_size=3,
                 scale_factor=2):
        super().__init__()

        self.compressed_channels = compressed_channels
        self.hr_channel_compressor = nn.Conv2d(hr_channels, self.compressed_channels, kernel_size=1)
        self.lr_channel_compressor = nn.Conv2d(lr_channels, self.compressed_channels, kernel_size=1)


        self.sampler = LocalSimGuidedSampler(in_channels=self.compressed_channels,
                                             scale=scale_factor,
                                             groups=feature_resample_group,
                                             kernel_size=kernel_size)

    def forward(self, hr_feat, lr_feat):

        compressed_hr_feat = self.hr_channel_compressor(hr_feat)
        compressed_lr_feat = self.lr_channel_compressor(lr_feat)


        downsampled_hr_feat = F.interpolate(compressed_hr_feat, size=compressed_lr_feat.shape[2:], mode='bilinear',
                                            align_corners=False)


        fused_feat = self.sampler(downsampled_hr_feat, compressed_lr_feat, compressed_lr_feat)

        return fused_feat


class LocalSimGuidedSampler(nn.Module):
    
    def __init__(self, in_channels, scale=2, groups=4, kernel_size=3, sim_type='cos', norm=True):
        super().__init__()
        self.scale = scale
        self.groups = groups
        self.sim_type = sim_type

        self.offset = nn.Conv2d(in_channels, 2 * groups * scale ** 2, kernel_size=kernel_size, padding=kernel_size // 2)
        self.direct_scale = nn.Conv2d(in_channels, 2 * groups * scale ** 2, kernel_size=kernel_size,
                                      padding=kernel_size // 2)

        self.norm = norm
        if self.norm:
            self.norm_hr = nn.GroupNorm(in_channels // 8, in_channels)
            self.norm_lr = nn.GroupNorm(in_channels // 8, in_channels)
        else:
            self.norm_hr = nn.Identity()
            self.norm_lr = nn.Identity()

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])).transpose(1, 2).unsqueeze(1).unsqueeze(0).to(
            x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(B, 2, -1, self.scale * H, self.scale * W)
        coords = coords.permute(0, 2, 3, 4, 1).flatten(0, 1)

        return F.grid_sample(x.view(B * self.groups, -1, x.size(-2), x.size(-1)), coords, mode='bilinear',
                             padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward(self, hr_x, lr_x, feat2sample):
        hr_x = self.norm_hr(hr_x)
        lr_x = self.norm_lr(lr_x)


        offset = (self.offset(lr_x) + self.direct_scale(hr_x)).sigmoid()
        return self.sample(feat2sample, offset)


if __name__ == '__main__':
    hr_feat = torch.rand(7, 512, 64, 64)
    lr_feat = torch.rand(7, 512, 32, 32)
    fusion = FreqFusion(hr_channels=512, lr_channels=512)
    fused_feat = fusion(hr_feat, lr_feat)
    print(f"Fused feature shape: {fused_feat.shape}")
