"""Spatial upscaler: a small U-Net operating at input resolution, followed by
a pixel-shuffle head that produces the final 2x-upsampled output.

Architecture reading of Spec 2 step 1 ("input res -> down -> down -> down ->
bottleneck -> up -> up -> up -> output res"): the down/up path is symmetric
and returns to *input* resolution (3 encoder downsamples, 3 decoder upsamples
with skip connections) -- the spec's "output res" at the end of that chain is
the point where the U-Net's own resolution matches its input again, not yet
the final 2x output. The actual 2x upscale is a dedicated pixel-shuffle head
after the U-Net. This is a judgement call (the spec's wording is ambiguous
between "output res = 2x input" reached directly by the down/up chain, vs.
the U-Net staying at input res with a separate upsample head) -- documented
here rather than silently assumed; see docs/PHASE-2-SUMMARY.md. Chosen because
it's the standard, parameter-efficient pattern for SR networks and keeps the
skip connections at matching (not mismatched) resolutions.

No normalisation layers (no BatchNorm/GroupNorm): common practice for SR
networks since normalisation can suppress fine detail, and it avoids
running-stats/eval-mode headaches for later ONNX export and hand-written WGSL
inference (Spec 4).
"""

import torch
import torch.nn as nn


class ConvAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class DownBlock(nn.Module):
    """Strided conv downsample (x0.5) followed by a same-resolution refine conv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = ConvAct(in_ch, out_ch, stride=2)
        self.refine = ConvAct(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.down(x))


class UpBlock(nn.Module):
    """Nearest-neighbour x2 upsample + conv, concat with the matching-resolution
    skip connection, then two refine convs back down to out_ch channels.

    Nearest-upsample-then-conv (not transposed conv) to avoid checkerboard
    artifacts -- a well-known failure mode of transposed convolutions in
    upsampling networks."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = ConvAct(in_ch + skip_ch, out_ch)
        self.conv2 = ConvAct(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        return self.conv2(x)


class SpatialUNet(nn.Module):
    """~0.92M params at the default channel widths.

    Input: (B, 4, H, W) -- jittered low-res colour (3ch) + linear depth (1ch).
    Output: (B, 3, 2H, 2W) -- upsampled colour.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        base_channels: int = 28,
    ):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 3, base_channels * 4

        self.stem = ConvAct(in_channels, c1)

        self.down1 = DownBlock(c1, c2)
        self.down2 = DownBlock(c2, c3)
        self.down3 = DownBlock(c3, c4)

        self.bottleneck = nn.Sequential(ConvAct(c4, c4), ConvAct(c4, c4))

        self.up1 = UpBlock(c4, c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up3 = UpBlock(c2, c1, c1)

        # Pixel-shuffle head: conv to out_channels * 4, then rearrange to 2x
        # spatial resolution. This is the dedicated upsample step (see module
        # docstring) -- the U-Net above stays at input resolution throughout.
        self.head = nn.Conv2d(c1, out_channels * 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)

        b = self.bottleneck(s3)

        u1 = self.up1(b, s2)
        u2 = self.up2(u1, s1)
        u3 = self.up3(u2, s0)

        out = self.head(u3)
        return self.pixel_shuffle(out)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = SpatialUNet()
    n_params = count_params(model)
    print(f"params: {n_params:,}")

    x = torch.randn(2, 4, 128, 128)
    y = model(x)
    print(f"input:  {tuple(x.shape)}")
    print(f"output: {tuple(y.shape)}")
    assert y.shape == (2, 3, 256, 256), "output must be exactly 2x input spatial resolution"
    print("shape check OK")
