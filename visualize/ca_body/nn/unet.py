"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch as th
import torch.nn as nn
import visualize.ca_body.nn.layers as la

from visualize.ca_body.nn.blocks import weights_initializer
from visualize.ca_body.nn.layers import Conv2dWNUB, ConvTranspose2dWNUB, glorot


class UNetWB(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        size: int,
        n_init_ftrs: int = 8,
        out_scale: float = 0.1,
    ):
        # super().__init__(*args, **kwargs)
        super().__init__()

        self.out_scale = out_scale

        F = n_init_ftrs

        self.size = size

        self.down1 = nn.Sequential(
            Conv2dWNUB(in_channels, F, self.size // 2, self.size // 2, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.down2 = nn.Sequential(
            Conv2dWNUB(F, 2 * F, self.size // 4, self.size // 4, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.down3 = nn.Sequential(
            Conv2dWNUB(2 * F, 4 * F, self.size // 8, self.size // 8, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.down4 = nn.Sequential(
            Conv2dWNUB(4 * F, 8 * F, self.size // 16, self.size // 16, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.down5 = nn.Sequential(
            Conv2dWNUB(8 * F, 16 * F, self.size // 32, self.size // 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.up1 = nn.Sequential(
            ConvTranspose2dWNUB(
                16 * F, 8 * F, self.size // 16, self.size // 16, 4, 2, 1
            ),
            nn.LeakyReLU(0.2),
        )
        self.up2 = nn.Sequential(
            ConvTranspose2dWNUB(8 * F, 4 * F, self.size // 8, self.size // 8, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.up3 = nn.Sequential(
            ConvTranspose2dWNUB(4 * F, 2 * F, self.size // 4, self.size // 4, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.up4 = nn.Sequential(
            ConvTranspose2dWNUB(2 * F, F, self.size // 2, self.size // 2, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.up5 = nn.Sequential(
            ConvTranspose2dWNUB(F, F, self.size, self.size, 4, 2, 1), nn.LeakyReLU(0.2)
        )
        self.out = Conv2dWNUB(
            F + in_channels, out_channels, self.size, self.size, kernel_size=1
        )
        self.apply(lambda x: glorot(x, 0.2))
        glorot(self.out, 1.0)

    def forward(self, x):
        x1 = x
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        # TODO: switch to concat?
        x = self.up1(x6) + x5
        x = self.up2(x) + x4
        x = self.up3(x) + x3
        x = self.up4(x) + x2
        x = self.up5(x)
        x = th.cat([x, x1], dim=1)
        return self.out(x) * self.out_scale


class UNetWBConcat(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        size: int,
        n_init_ftrs: int = 8,
    ):
        super().__init__()

        F = n_init_ftrs

        self.size = size

        self.down1 = nn.Sequential(
            la.Conv2dWNUB(in_channels, F, self.size // 2, self.size // 2, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.down2 = nn.Sequential(
            la.Conv2dWNUB(F, 2 * F, self.size // 4, self.size // 4, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.down3 = nn.Sequential(
            la.Conv2dWNUB(2 * F, 4 * F, self.size // 8, self.size // 8, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.down4 = nn.Sequential(
            la.Conv2dWNUB(4 * F, 8 * F, self.size // 16, self.size // 16, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.down5 = nn.Sequential(
            la.Conv2dWNUB(8 * F, 16 * F, self.size // 32, self.size // 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.up1 = nn.Sequential(
            la.ConvTranspose2dWNUB(
                16 * F, 8 * F, self.size // 16, self.size // 16, 4, 2, 1
            ),
            nn.LeakyReLU(0.2),
        )
        self.up2 = nn.Sequential(
            la.ConvTranspose2dWNUB(
                2 * 8 * F, 4 * F, self.size // 8, self.size // 8, 4, 2, 1
            ),
            nn.LeakyReLU(0.2),
        )
        self.up3 = nn.Sequential(
            la.ConvTranspose2dWNUB(
                2 * 4 * F, 2 * F, self.size // 4, self.size // 4, 4, 2, 1
            ),
            nn.LeakyReLU(0.2),
        )
        self.up4 = nn.Sequential(
            la.ConvTranspose2dWNUB(
                2 * 2 * F, F, self.size // 2, self.size // 2, 4, 2, 1
            ),
            nn.LeakyReLU(0.2),
        )
        self.up5 = nn.Sequential(
            la.ConvTranspose2dWNUB(2 * F, F, self.size, self.size, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.out = la.Conv2dWNUB(
            F + in_channels, out_channels, self.size, self.size, kernel_size=1
        )
        self.apply(lambda x: la.glorot(x, 0.2))
        la.glorot(self.out, 1.0)

    def forward(self, x):
        x1 = x
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = th.cat([self.up1(x6), x5], 1)
        x = th.cat([self.up2(x), x4], 1)
        x = th.cat([self.up3(x), x3], 1)
        x = th.cat([self.up4(x), x2], 1)
        x = self.up5(x)
        x = th.cat([x, x1], dim=1)
        return self.out(x)


class UNetW(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_init_ftrs,
        kernel_size=4,
        out_scale=1.0,
    ):
        super().__init__()

        self.out_scale = out_scale

        F = n_init_ftrs

        self.down1 = nn.Sequential(
            la.Conv2dWN(in_channels, F, kernel_size, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.down2 = nn.Sequential(
            la.Conv2dWN(F, 2 * F, kernel_size, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.down3 = nn.Sequential(
            la.Conv2dWN(2 * F, 4 * F, kernel_size, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.down4 = nn.Sequential(
            la.Conv2dWN(4 * F, 8 * F, kernel_size, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.down5 = nn.Sequential(
            la.Conv2dWN(8 * F, 16 * F, kernel_size, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.up1 = nn.Sequential(
            la.ConvTranspose2dWN(16 * F, 8 * F, kernel_size, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.up2 = nn.Sequential(
            la.ConvTranspose2dWN(8 * F, 4 * F, kernel_size, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.up3 = nn.Sequential(
            la.ConvTranspose2dWN(4 * F, 2 * F, kernel_size, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.up4 = nn.Sequential(
            la.ConvTranspose2dWN(2 * F, F, kernel_size, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.up5 = nn.Sequential(
            la.ConvTranspose2dWN(F, F, kernel_size, 2, 1), nn.LeakyReLU(0.2)
        )
        self.out = la.Conv2dWN(F + in_channels, out_channels, kernel_size=1)
        self.apply(weights_initializer(0.2))
        self.out.apply(weights_initializer(1.0))

    def forward(self, x):
        x1 = x
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        # TODO: switch to concat?
        x = self.up1(x6) + x5
        x = self.up2(x) + x4
        x = self.up3(x) + x3
        x = self.up4(x) + x2
        x = self.up5(x)
        x = th.cat([x, x1], dim=1)
        return self.out(x) * self.out_scale
