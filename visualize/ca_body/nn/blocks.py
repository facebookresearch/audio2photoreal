"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


import logging
from turtle import forward

import visualize.ca_body.nn.layers as la
from visualize.ca_body.nn.layers import weight_norm_wrapper

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


# pyre-ignore
def weights_initializer(lrelu_slope=0.2):
    # pyre-ignore
    def init_fn(m):
        if isinstance(
            m,
            (
                nn.Conv2d,
                nn.Conv1d,
                nn.ConvTranspose2d,
                nn.Linear,
            ),
        ):
            gain = nn.init.calculate_gain("leaky_relu", lrelu_slope)
            nn.init.kaiming_uniform_(m.weight.data, a=gain)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias.data)
        else:
            logger.debug(f"skipping initialization for {m}")

    return init_fn


# pyre-ignore
def WeightNorm(x, dim=0):
    return nn.utils.weight_norm(x, dim=dim)


# pyre-ignore
def np_warp_bias(uv_size):
    xgrid, ygrid = np.meshgrid(np.linspace(-1.0, 1.0, uv_size), np.linspace(-1.0, 1.0, uv_size))
    grid = np.concatenate((xgrid[None, :, :], ygrid[None, :, :]), axis=0)[None, ...].astype(
        np.float32
    )
    return grid


class Conv2dBias(nn.Conv2d):
    __annotations__ = {"bias": th.Tensor}

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        size,
        stride=1,
        padding=1,
        bias=True,
        *args,
        **kwargs,
    ):
        super().__init__(
            in_channels,
            out_channels,
            bias=False,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            *args,
            **kwargs,
        )
        if not bias:
            logger.warning("ignoring bias=False")
        self.bias = nn.Parameter(th.zeros(out_channels, size, size))

    def forward(self, x):
        bias = self.bias.clone()
        return (
            # pyre-ignore
            th.conv2d(
                x,
                self.weight,
                bias=None,
                stride=self.stride,
                # pyre-ignore
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            + bias[np.newaxis]
        )


class Conv1dBias(nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        size,
        stride=1,
        padding=0,
        bias=True,
        *args,
        **kwargs,
    ):
        super().__init__(
            in_channels,
            out_channels,
            bias=False,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            *args,
            **kwargs,
        )
        if not bias:
            logger.warning("ignoring bias=False")
        self.bias = nn.Parameter(th.zeros(out_channels, size))

    def forward(self, x):
        return (
            # pyre-ignore
            th.conv1d(
                x,
                self.weight,
                bias=None,
                stride=self.stride,
                # pyre-ignore
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            + self.bias
        )


class UpConvBlock(nn.Module):
    # pyre-ignore
    def __init__(self, in_channels, out_channels, size, lrelu_slope=0.2):
        super().__init__()
        # Intergration: it was not exist in github, but assume upsample is same as other class
        self.upsample = nn.UpsamplingBilinear2d(size)
        self.conv_resize = la.Conv2dWN(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1
        )
        self.conv1 = la.Conv2dWNUB(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            height=size,
            width=size,
            padding=1,
        )
        self.lrelu1 = nn.LeakyReLU(lrelu_slope)
        # self.conv2 = nn.utils.weight_norm(
        #     Conv2dBias(in_channels, out_channels, kernel_size=3, size=size), dim=None,
        # )
        # self.lrelu2 = nn.LeakyReLU(lrelu_slope)

    # pyre-ignore
    def forward(self, x):
        x_up = self.upsample(x)
        x_skip = self.conv_resize(x_up)
        x = self.conv1(x_up)
        x = self.lrelu1(x)
        return x + x_skip


class ConvBlock1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        size,
        lrelu_slope=0.2,
        kernel_size=3,
        padding=1,
        wnorm_dim=0,
    ):
        super().__init__()

        self.conv_resize = WeightNorm(
            nn.Conv1d(in_channels, out_channels, kernel_size=1), dim=wnorm_dim
        )
        self.conv1 = WeightNorm(
            Conv1dBias(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=padding,
                size=size,
            ),
            dim=wnorm_dim,
        )
        self.lrelu1 = nn.LeakyReLU(lrelu_slope)
        self.conv2 = WeightNorm(
            Conv1dBias(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                size=size,
            ),
            dim=wnorm_dim,
        )
        self.lrelu2 = nn.LeakyReLU(lrelu_slope)

    def forward(self, x):
        x_skip = self.conv_resize(x)
        x = self.conv1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.lrelu2(x)
        return x + x_skip


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        size,
        lrelu_slope=0.2,
        kernel_size=3,
        padding=1,
        wnorm_dim=0,
    ):
        super().__init__()

        Conv2dWNUB = weight_norm_wrapper(la.Conv2dUB, "Conv2dWNUB", g_dim=wnorm_dim, v_dim=None)
        Conv2dWN = weight_norm_wrapper(th.nn.Conv2d, "Conv2dWN", g_dim=wnorm_dim, v_dim=None)

        # TODO: do we really need this?
        self.conv_resize = Conv2dWN(in_channels, out_channels, kernel_size=1)
        self.conv1 = Conv2dWNUB(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            height=size,
            width=size,
        )

        self.lrelu1 = nn.LeakyReLU(lrelu_slope)
        self.conv2 = Conv2dWNUB(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            height=size,
            width=size,
        )
        self.lrelu2 = nn.LeakyReLU(lrelu_slope)

    def forward(self, x):
        x_skip = self.conv_resize(x)
        x = self.conv1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.lrelu2(x)
        return x + x_skip


class ConvBlockNoSkip(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        size,
        lrelu_slope=0.2,
        kernel_size=3,
        padding=1,
        wnorm_dim=0,
    ):
        super().__init__()

        self.conv1 = WeightNorm(
            Conv2dBias(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=padding,
                size=size,
            ),
            dim=wnorm_dim,
        )
        self.lrelu1 = nn.LeakyReLU(lrelu_slope)
        self.conv2 = WeightNorm(
            Conv2dBias(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                size=size,
            ),
            dim=wnorm_dim,
        )
        self.lrelu2 = nn.LeakyReLU(lrelu_slope)

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.lrelu2(x)
        return x


class ConvDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size, lrelu_slope=0.2, groups=1, wnorm_dim=0):
        """Constructor.

        Args:
            in_channels: int, # of input channels
            out_channels: int, # of input channels
            size: the *input* size
        """
        super().__init__()

        Conv2dWNUB = weight_norm_wrapper(la.Conv2dUB, "Conv2dWNUB", g_dim=wnorm_dim, v_dim=None)
        Conv2dWN = weight_norm_wrapper(th.nn.Conv2d, "Conv2dWN", g_dim=wnorm_dim, v_dim=None)

        self.conv_resize = Conv2dWN(
            in_channels, out_channels, kernel_size=1, stride=2, groups=groups
        )
        self.conv1 = Conv2dWNUB(
            in_channels,
            in_channels,
            kernel_size=3,
            height=size,
            width=size,
            groups=groups,
            padding=1,
        )
        self.lrelu1 = nn.LeakyReLU(lrelu_slope)

        self.conv2 = Conv2dWNUB(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            height=size // 2,
            width=size // 2,
            groups=groups,
            padding=1,
        )
        self.lrelu2 = nn.LeakyReLU(lrelu_slope)

    def forward(self, x):
        x_skip = self.conv_resize(x)
        x = self.conv1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.lrelu2(x)
        return x + x_skip


class UpConvBlockDeep(nn.Module):
    def __init__(self, in_channels, out_channels, size, lrelu_slope=0.2, wnorm_dim=0, groups=1):
        super().__init__()
        self.upsample = nn.UpsamplingBilinear2d(size)

        Conv2dWNUB = weight_norm_wrapper(la.Conv2dUB, "Conv2dWNUB", g_dim=wnorm_dim, v_dim=None)
        Conv2dWN = weight_norm_wrapper(th.nn.Conv2d, "Conv2dWN", g_dim=wnorm_dim, v_dim=None)
        # NOTE: the old one normalizes only across one dimension

        self.conv_resize = Conv2dWN(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            groups=groups,
        )
        self.conv1 = Conv2dWNUB(
            in_channels,
            in_channels,
            kernel_size=3,
            height=size,
            width=size,
            padding=1,
            groups=groups,
        )
        self.lrelu1 = nn.LeakyReLU(lrelu_slope)
        self.conv2 = Conv2dWNUB(
            in_channels,
            out_channels,
            kernel_size=3,
            height=size,
            width=size,
            padding=1,
            groups=groups,
        )
        self.lrelu2 = nn.LeakyReLU(lrelu_slope)

    def forward(self, x):
        x_up = self.upsample(x)
        x_skip = self.conv_resize(x_up)

        x = x_up
        x = self.conv1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.lrelu2(x)

        return x + x_skip


class ConvBlockPositional(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        pos_map,
        lrelu_slope=0.2,
        kernel_size=3,
        padding=1,
        wnorm_dim=0,
    ):
        """Block with positional encoding.

        Args:
            in_channels: # of input channels (not counting the positional encoding)
            out_channels: # of output channels
            pos_map: tensor [P, size, size]
        """
        super().__init__()
        assert len(pos_map.shape) == 3 and pos_map.shape[1] == pos_map.shape[2]
        self.register_buffer("pos_map", pos_map)

        self.conv_resize = WeightNorm(nn.Conv2d(in_channels, out_channels, 1), dim=wnorm_dim)

        self.conv1 = WeightNorm(
            nn.Conv2d(
                in_channels + pos_map.shape[0],
                in_channels,
                kernel_size=3,
                padding=padding,
            ),
            dim=wnorm_dim,
        )
        self.lrelu1 = nn.LeakyReLU(lrelu_slope)
        self.conv2 = WeightNorm(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding),
            dim=wnorm_dim,
        )
        self.lrelu2 = nn.LeakyReLU(lrelu_slope)

    def forward(self, x):
        B = x.shape[0]

        x_skip = self.conv_resize(x)

        pos = self.pos_map[np.newaxis].expand(B, -1, -1, -1)

        x = th.cat([x, pos], dim=1)
        x = self.conv1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.lrelu2(x)
        return x + x_skip


class UpConvBlockPositional(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        pos_map,
        lrelu_slope=0.2,
        wnorm_dim=0,
    ):
        """Block with positional encoding.

        Args:
            in_channels: # of input channels (not counting the positional encoding)
            out_channels: # of output channels
            pos_map: tensor [P, size, size]
        """
        super().__init__()
        assert len(pos_map.shape) == 3 and pos_map.shape[1] == pos_map.shape[2]
        self.register_buffer("pos_map", pos_map)
        size = pos_map.shape[1]

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.upsample = nn.UpsamplingBilinear2d(size)

        if in_channels != out_channels:
            self.conv_resize = WeightNorm(nn.Conv2d(in_channels, out_channels, 1), dim=wnorm_dim)

        self.conv1 = WeightNorm(
            nn.Conv2d(
                in_channels + pos_map.shape[0],
                in_channels,
                kernel_size=3,
                padding=1,
            ),
            dim=wnorm_dim,
        )
        self.lrelu1 = nn.LeakyReLU(lrelu_slope)
        self.conv2 = WeightNorm(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            dim=wnorm_dim,
        )
        self.lrelu2 = nn.LeakyReLU(lrelu_slope)

    def forward(self, x):
        B = x.shape[0]

        x_up = self.upsample(x)

        x_skip = x_up
        if self.in_channels != self.out_channels:
            x_skip = self.conv_resize(x_up)

        pos = self.pos_map[np.newaxis].expand(B, -1, -1, -1)

        x = th.cat([x_up, pos], dim=1)
        x = self.conv1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.lrelu2(x)

        return x + x_skip


class UpConvBlockDeepNoBias(nn.Module):
    def __init__(self, in_channels, out_channels, size, lrelu_slope=0.2, wnorm_dim=0, groups=1):
        super().__init__()
        self.upsample = nn.UpsamplingBilinear2d(size)
        # NOTE: the old one normalizes only across one dimension
        self.conv_resize = WeightNorm(
            nn.Conv2d(in_channels, out_channels, 1, groups=groups), dim=wnorm_dim
        )
        self.conv1 = WeightNorm(
            nn.Conv2d(in_channels, in_channels, padding=1, kernel_size=3, groups=groups),
            dim=wnorm_dim,
        )
        self.lrelu1 = nn.LeakyReLU(lrelu_slope)
        self.conv2 = WeightNorm(
            nn.Conv2d(in_channels, out_channels, padding=1, kernel_size=3, groups=groups),
            dim=wnorm_dim,
        )
        self.lrelu2 = nn.LeakyReLU(lrelu_slope)

    def forward(self, x):
        x_up = self.upsample(x)
        x_skip = self.conv_resize(x_up)

        x = x_up
        x = self.conv1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.lrelu2(x)

        return x + x_skip


class UpConvBlockXDeep(nn.Module):
    def __init__(self, in_channels, out_channels, size, lrelu_slope=0.2, wnorm_dim=0):
        super().__init__()
        self.upsample = nn.UpsamplingBilinear2d(size)
        # TODO: see if this is necce
        self.conv_resize = WeightNorm(nn.Conv2d(in_channels, out_channels, 1), dim=wnorm_dim)
        self.conv1 = WeightNorm(
            Conv2dBias(in_channels, in_channels // 2, kernel_size=3, size=size),
            dim=wnorm_dim,
        )
        self.lrelu1 = nn.LeakyReLU(lrelu_slope)

        self.conv2 = WeightNorm(
            Conv2dBias(in_channels // 2, in_channels // 2, kernel_size=3, size=size),
            dim=wnorm_dim,
        )
        self.lrelu2 = nn.LeakyReLU(lrelu_slope)

        self.conv2 = WeightNorm(
            Conv2dBias(in_channels // 2, in_channels // 2, kernel_size=3, size=size),
            dim=wnorm_dim,
        )
        self.lrelu2 = nn.LeakyReLU(lrelu_slope)

        self.conv3 = WeightNorm(
            Conv2dBias(in_channels // 2, out_channels, kernel_size=3, size=size),
            dim=wnorm_dim,
        )
        self.lrelu3 = nn.LeakyReLU(lrelu_slope)

    def forward(self, x):
        x_up = self.upsample(x)
        x_skip = self.conv_resize(x_up)

        x = x_up
        x = self.conv1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.lrelu2(x)
        x = self.conv3(x)
        x = self.lrelu3(x)

        return x + x_skip


class UpConvCondBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size, cond_channels, lrelu_slope=0.2):
        super().__init__()
        self.upsample = nn.UpsamplingBilinear2d(size)
        self.conv_resize = nn.utils.weight_norm(nn.Conv2d(in_channels, out_channels, 1), dim=None)
        self.conv1 = WeightNorm(
            Conv2dBias(in_channels + cond_channels, in_channels, kernel_size=3, size=size),
        )
        self.lrelu1 = nn.LeakyReLU(lrelu_slope)
        self.conv2 = WeightNorm(
            Conv2dBias(in_channels, out_channels, kernel_size=3, size=size),
        )
        self.lrelu2 = nn.LeakyReLU(lrelu_slope)

    def forward(self, x, cond):
        x_up = self.upsample(x)
        x_skip = self.conv_resize(x_up)

        x = x_up
        x = th.cat([x, cond], dim=1)
        x = self.conv1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.lrelu2(x)

        return x + x_skip


class UpConvBlockPS(nn.Module):
    # pyre-ignore
    def __init__(self, n_in, n_out, size, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = la.Conv2dWNUB(
            n_in,
            n_out * 4,
            size,
            size,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.ps = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.lrelu(x)
        return self.ps(x)


# pyre-ignore
def apply_crop(
    image,
    ymin,
    ymax,
    xmin,
    xmax,
):
    """Crops a region from an image."""
    # NOTE: here we are expecting one of [H, W] [H, W, C] [B, H, W, C]
    if len(image.shape) == 2:
        return image[ymin:ymax, xmin:xmax]
    elif len(image.shape) == 3:
        return image[ymin:ymax, xmin:xmax, :]
    elif len(image.shape) == 4:
        return image[:, ymin:ymax, xmin:xmax, :]
    else:
        raise ValueError("provide a batch of images or a single image")


def tile1d(x, size):
    """Tile a given set of features into a convolutional map.
    Args:
        x: float tensor of shape [N, F]
        size: int or a tuple
    Returns:
        a feature map [N, F, ∑size[0], size[1]]
    """
    # size = size if isinstance(size, tuple) else (size, size)
    return x[:, :, np.newaxis].expand(-1, -1, size)


def tile2d(x, size: int):
    """Tile a given set of features into a convolutional map.

    Args:
        x: float tensor of shape [N, F]
        size: int or a tuple

    Returns:
        a feature map [N, F, size[0], size[1]]
    """
    # size = size if isinstance(size, tuple) else (size, size)
    # NOTE: expecting only int here (!!!)
    return x[:, :, np.newaxis, np.newaxis].expand(-1, -1, size, size)


def sample_negative_idxs(size, *args, **kwargs):
    idxs = th.randperm(size, *args, **kwargs)
    if th.all(idxs == th.arange(size, dtype=idxs.dtype, device=idxs.device)):
        return th.flip(idxs, (0,))
    return idxs


def icnr_init(x, scale=2, init=nn.init.kaiming_normal_):
    ni, nf, h, w = x.shape
    ni2 = int(ni / (scale**2))
    k = init(x.new_zeros([ni2, nf, h, w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    return k.contiguous().view([nf, ni, h, w]).transpose(0, 1)


class PixelShuffleWN(nn.Module):
    """PixelShuffle with the right initialization.

    NOTE: make sure to create this one
    """

    def __init__(self, n_in, n_out, upscale_factor=2):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.n_in = n_in
        self.n_out = n_out
        self.conv = la.Conv2dWN(n_in, n_out * (upscale_factor**2), kernel_size=1, padding=0)
        # NOTE: the bias is 2K?
        self.ps = nn.PixelShuffle(upscale_factor)
        self._init_icnr()

    def _init_icnr(self):
        self.conv.weight_v.data.copy_(icnr_init(self.conv.weight_v.data))
        self.conv.weight_g.data.copy_(
            ((self.conv.weight_v.data**2).sum(dim=[1, 2, 3]) ** 0.5)[:, None, None, None]
        )

    def forward(self, x):
        x = self.conv(x)
        return self.ps(x)


class UpscaleNet(nn.Module):
    def __init__(self, in_channels, out_channels=3, n_ftrs=16, size=1024, upscale_factor=2):
        super().__init__()

        self.conv_block = nn.Sequential(
            la.Conv2dWNUB(in_channels, n_ftrs, size, size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            la.Conv2dWNUB(n_ftrs, n_ftrs, size, size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.out_block = la.Conv2dWNUB(
            n_ftrs,
            out_channels * upscale_factor**2,
            size,
            size,
            kernel_size=1,
            padding=0,
        )

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=upscale_factor)
        self.apply(lambda x: la.glorot(x, 0.2))
        self.out_block.apply(weights_initializer(1.0))

    def forward(self, x):
        x = self.conv_block(x)
        x = self.out_block(x)
        return self.pixel_shuffle(x)


