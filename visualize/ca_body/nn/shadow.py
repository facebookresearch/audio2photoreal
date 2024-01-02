"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import logging

from typing import Optional, Dict


import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# TODO: use shared utils here?
import visualize.ca_body.nn.layers as la
from visualize.ca_body.nn.blocks import tile2d, weights_initializer

logger = logging.getLogger(__name__)


class ShadowUNet(nn.Module):
    def __init__(
        self,
        uv_size,
        ao_mean,
        shadow_size,
        lrelu_slope=0.2,
        beta=1.0,
        n_dims=64,
        interp_mode="bilinear",
        biases=True,
        trainable_mean=False,
    ):
        super().__init__()

        # this is the size of the output
        self.uv_size = uv_size
        self.shadow_size = shadow_size

        ao_mean = F.interpolate(
            th.as_tensor(ao_mean)[np.newaxis],
            size=(self.shadow_size, self.shadow_size),
        )[0]
        if not trainable_mean:
            # TODO:
            self.register_buffer("ao_mean", ao_mean)
        else:
            self.register_parameter("ao_mean", th.nn.Parameter(ao_mean))

        self.depth = 3
        self.lrelu_slope = lrelu_slope
        self.interp_mode = interp_mode
        self.align_corners = None
        if interp_mode == "bilinear":
            self.align_corners = False

        # the base number of dimensions for the shadow maps
        n_dims = n_dims

        # TODO: generate this?
        self.n_enc_dims = [
            (1, n_dims),
            (n_dims, n_dims),
            (n_dims, n_dims),
            (n_dims, n_dims),
        ]

        self.sizes = [shadow_size // (2**i) for i in range(len(self.n_enc_dims))]

        logger.debug(f"sizes: {self.sizes}")

        self.enc_layers = nn.ModuleList()
        for i, size in enumerate(self.sizes):
            n_in, n_out = self.n_enc_dims[i]
            logger.debug(f"EncoderLayers({i}): {n_in}, {n_out}, {size}")
            self.enc_layers.append(
                nn.Sequential(
                    la.Conv2dWNUB(
                        n_in,
                        n_out,
                        kernel_size=3,
                        height=size,
                        width=size,
                        stride=1,
                        padding=1,
                    ),
                    nn.LeakyReLU(self.lrelu_slope, inplace=True),
                )
            )

        self.n_dec_dims = [
            (n_dims, n_dims),
            (n_dims * 2, n_dims),
            (n_dims * 2, n_dims),
            (n_dims * 2, n_dims),
        ]
        self.dec_layers = nn.ModuleList()
        for i in range(len(self.sizes)):
            size = self.sizes[-i - 1]
            n_in, n_out = self.n_dec_dims[i]
            logger.debug(f"DecoderLayer({i}): {n_in}, {n_out}, {size}")

            self.dec_layers.append(
                nn.Sequential(
                    la.Conv2dWNUB(
                        n_in,
                        n_out,
                        kernel_size=3,
                        height=size,
                        width=size,
                        stride=1,
                        padding=1,
                    ),
                    nn.LeakyReLU(self.lrelu_slope, inplace=True),
                )
            )

        self.apply(weights_initializer(self.lrelu_slope))

        if biases:
            self.shadow_pred = la.Conv2dWNUB(
                self.n_dec_dims[-1][-1],
                1,
                kernel_size=3,
                height=self.sizes[0],
                width=self.sizes[0],
                stride=1,
                padding=1,
            )
        else:
            self.shadow_pred = la.Conv2dWN(
                self.n_dec_dims[-1][-1],
                1,
                kernel_size=3,
                stride=1,
                padding=1,
            )

        self.shadow_pred.apply(weights_initializer(1.0))
        self.beta = beta

    def forward(self, ao_map):
        # resizing the inputs if necessary
        if ao_map.shape[-2:] != (self.shadow_size, self.shadow_size):
            ao_map = F.interpolate(ao_map, size=(self.shadow_size, self.shadow_size))

        x = ao_map - self.ao_mean

        enc_acts = []
        # unet enc
        for i, layer in enumerate(self.enc_layers):
            # TODO: try applying a 1D sparse op?
            x = layer(x)
            enc_acts.append(x)
            # TODO: add this layer elsewhere?
            if i < len(self.sizes) - 1:
                x = F.interpolate(
                    x,
                    scale_factor=0.5,
                    mode="bilinear",
                    recompute_scale_factor=True,
                    align_corners=True,
                )

        # we do not need the last one?
        for i, layer in enumerate(self.dec_layers):
            if i > 0:
                x_prev = enc_acts[-i - 1]
                x = F.interpolate(x, size=x_prev.shape[2:4], mode="bilinear", align_corners=True)
                x = th.cat([x, x_prev], dim=1)
            x = layer(x)

        shadow_map_lowres = th.sigmoid(self.shadow_pred(x) + self.beta)
        shadow_map = F.interpolate(
            shadow_map_lowres,
            (self.uv_size, self.uv_size),
            mode=self.interp_mode,
            align_corners=self.align_corners,
        )

        return {
            "shadow_map": shadow_map,
            "ao_map": ao_map,
            "shadow_map_lowres": shadow_map_lowres,
        }


class FloorShadowDecoder(nn.Module):
    def __init__(
        self,
        uv_size,
        beta=1.0,
    ):
        super().__init__()

        # TODO: can we reduce # dims here?
        self.down1 = nn.Sequential(la.Conv2dWNUB(1, 64, 256, 256, 4, 2, 1), nn.LeakyReLU(0.2))
        self.down2 = nn.Sequential(la.Conv2dWNUB(64, 64, 128, 128, 4, 2, 1), nn.LeakyReLU(0.2))
        self.down3 = nn.Sequential(la.Conv2dWNUB(64, 128, 64, 64, 4, 2, 1), nn.LeakyReLU(0.2))
        self.down4 = nn.Sequential(la.Conv2dWNUB(128, 256, 32, 32, 4, 2, 1), nn.LeakyReLU(0.2))
        self.down5 = nn.Sequential(la.Conv2dWNUB(256, 512, 16, 16, 4, 2, 1), nn.LeakyReLU(0.2))
        self.up1 = nn.Sequential(
            la.ConvTranspose2dWNUB(512, 256, 32, 32, 4, 2, 1), nn.LeakyReLU(0.2)
        )
        self.up2 = nn.Sequential(
            la.ConvTranspose2dWNUB(256, 128, 64, 64, 4, 2, 1), nn.LeakyReLU(0.2)
        )
        self.up3 = nn.Sequential(
            la.ConvTranspose2dWNUB(128, 64, 128, 128, 4, 2, 1), nn.LeakyReLU(0.2)
        )
        self.up4 = nn.Sequential(
            la.ConvTranspose2dWNUB(64, 64, 256, 256, 4, 2, 1), nn.LeakyReLU(0.2)
        )
        self.up5 = nn.Sequential(la.ConvTranspose2dWNUB(64, 1, 512, 512, 4, 2, 1))

        self.uv_size = uv_size

        self.apply(lambda x: la.glorot(x, 0.2))
        la.glorot(self.up5, 1.0)

        self.beta = beta

    def forward(self, aomap: th.Tensor):
        aomap = F.interpolate(
            aomap,
            size=(self.uv_size, self.uv_size),
            mode="bilinear",
            align_corners=True,
        )

        x2 = self.down1(aomap - 0.5)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6) + x5
        x = self.up2(x) + x4
        x = self.up3(x) + x3
        x = self.up4(x) + x2
        logits = (th.tanh(self.up5(x) + aomap) + 1.0) / 2.0

        return {"shadow_map": logits}


class ShadowUNet_PoseCond(nn.Module):
    def __init__(
        self,
        uv_size,
        ao_mean,
        shadow_size,
        # uv_coords, # for bottleneck
        # uv_mapping, # for bottleneck
        # uv_faces, # for bottleneck
        lrelu_slope=0.2,
        beta=1.0,
        n_dims=64,
        n_pose_dims=6,  # root orientation only
        n_pose_enc_dims=32,
        interp_mode="bilinear",
    ):
        super().__init__()

        self.uv_size = uv_size

        self.register_buffer("ao_mean", th.as_tensor(ao_mean))

        # self.register_buffer("uv_coords", th.as_tensor(uv_coords))
        # self.register_buffer("uv_mapping", th.as_tensor(uv_mapping))
        # self.register_buffer("uv_faces", th.as_tensor(uv_faces))

        self.depth = 3
        self.lrelu_slope = lrelu_slope
        self.interp_mode = interp_mode

        self.uv_size = uv_size

        # the base number of dimensions for the shadow maps
        n_dims = n_dims

        # TODO: generate this?
        self.n_enc_dims = [
            (1, n_dims),
            (n_dims, n_dims),
            (n_dims, n_dims),
            (n_dims, n_dims),
        ]

        self.shadow_size = shadow_size
        self.sizes = [shadow_size // (2**i) for i in range(len(self.n_enc_dims))]

        logger.info(f" shadow map size: {self.shadow_size}")
        # logger.info(f"sizes: {self.sizes}")

        #####
        ## FC for root pose encoding
        self.num_pose_dims = n_pose_dims
        self.num_pose_enc_dims = n_pose_enc_dims
        self.pose_fc_block = nn.Sequential(
            la.LinearWN(self.num_pose_dims, self.num_pose_enc_dims),
            nn.LeakyReLU(lrelu_slope),
        )

        self.pose_conv_block = la.Conv2dWNUB(
            in_channels=self.num_pose_dims,
            out_channels=self.num_pose_enc_dims,
            kernel_size=3,
            height=self.sizes[-1],
            width=self.sizes[-1],
            padding=1,
        )

        self.enc_layers = nn.ModuleList()
        for i, size in enumerate(self.sizes):
            n_in, n_out = self.n_enc_dims[i]
            # logger.info(f"EncoderLayers({i}): {n_in}, {n_out}, {size}")
            self.enc_layers.append(
                nn.Sequential(
                    la.Conv2dWNUB(
                        n_in,
                        n_out,
                        kernel_size=3,
                        height=size,
                        width=size,
                        stride=1,
                        padding=1,
                    ),
                    nn.LeakyReLU(self.lrelu_slope, inplace=True),
                )
            )

        self.n_dec_dims = [
            (n_dims + self.num_pose_enc_dims, n_dims),
            (n_dims * 2, n_dims),
            (n_dims * 2, n_dims),
            (n_dims * 2, n_dims),
        ]
        self.dec_layers = nn.ModuleList()
        for i in range(len(self.sizes)):
            size = self.sizes[-i - 1]
            n_in, n_out = self.n_dec_dims[i]
            # logger.info(f"DecoderLayer({i}): {n_in}, {n_out}, {size}")
            self.dec_layers.append(
                nn.Sequential(
                    la.Conv2dWNUB(
                        n_in,
                        n_out,
                        kernel_size=3,
                        height=size,
                        width=size,
                        stride=1,
                        padding=1,
                    ),
                    nn.LeakyReLU(self.lrelu_slope, inplace=True),
                )
            )

        self.apply(weights_initializer(self.lrelu_slope))
        self.shadow_pred = la.Conv2dWNUB(
            self.n_dec_dims[-1][-1],
            1,
            kernel_size=3,
            height=self.sizes[0],
            width=self.sizes[0],
            stride=1,
            padding=1,
        )

        self.shadow_pred.apply(weights_initializer(1.0))
        self.beta = beta

    def forward(self, ao_map, pose_vec):
        # import pdb; pdb.set_trace()
        x = ao_map - self.ao_mean

        x = F.interpolate(x, size=(self.shadow_size, self.shadow_size))

        enc_acts = []
        # unet enc
        for i, layer in enumerate(self.enc_layers):
            # for i in range(len(self.sizes)):
            # TODO: try applying a 1D sparse op?
            # x = self.enc_layers[i](x)
            x = layer(x)
            enc_acts.append(x)
            # TODO: add this layer elsewhere?
            if i < len(self.sizes) - 1:
                x = F.interpolate(
                    x,
                    scale_factor=0.5,
                    mode="bilinear",
                    recompute_scale_factor=True,
                    align_corners=True,
                )

        pose_enc = self.pose_conv_block(tile2d(pose_vec, self.sizes[-1]))

        # we do not need the last one?
        x = th.cat([x, pose_enc], dim=1)

        for i, layer in enumerate(self.dec_layers):
            if i > 0:
                x_prev = enc_acts[-i - 1]
                x = F.interpolate(x, size=x_prev.shape[2:4], mode="bilinear", align_corners=True)
                x = th.cat([x, x_prev], dim=1)
            x = layer(x)

        shadow_map_lowres = th.sigmoid(self.shadow_pred(x) + self.beta)
        shadow_map = F.interpolate(
            shadow_map_lowres, (self.uv_size, self.uv_size), mode=self.interp_mode
        )
        return {"shadow_map": shadow_map, "ao_map": ao_map}


class PoseToShadow(nn.Module):
    def __init__(
        self,
        n_pose_dims,
        uv_size,
        beta=1.0,
    ) -> None:
        super().__init__()
        self.n_pose_dims = n_pose_dims
        self.uv_size = uv_size

        self.fc_block = nn.Sequential(
            la.LinearWN(self.n_pose_dims, 256 * 4 * 4),
            nn.LeakyReLU(0.2),
        )
        self.conv_block = nn.Sequential(
            la.ConvTranspose2dWNUB(256, 256, 8, 8, 4, 2, 1),
            nn.LeakyReLU(0.2),
            la.ConvTranspose2dWNUB(256, 128, 16, 16, 4, 2, 1),
            nn.LeakyReLU(0.2),
            la.ConvTranspose2dWNUB(128, 128, 32, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            la.ConvTranspose2dWNUB(128, 64, 64, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            # la.ConvTranspose2dWNUB(64, 64, 128, 128, 4, 2, 1),
            # nn.LeakyReLU(0.2),
            # la.ConvTranspose2dWNUB(64, 1, 256, 256, 4, 2, 1),
            la.ConvTranspose2dWNUB(64, 1, 128, 128, 4, 2, 1),
        )
        self.beta = beta
        self.apply(lambda x: la.glorot(x, 0.2))
        la.glorot(self.conv_block[-1], 1.0)

    def forward(self, pose: th.Tensor):
        assert pose.shape
        x = self.fc_block(pose)
        x = self.conv_block(x.reshape(-1, 256, 4, 4))
        shadow_map_lowres = th.sigmoid(x + self.beta)

        shadow_map = F.interpolate(
            shadow_map_lowres, size=(self.uv_size, self.uv_size), mode="bilinear"
        )
        return {"shadow_map": shadow_map}


class DistMapShadowUNet(nn.Module):
    def __init__(
        self,
        uv_size,
        shadow_size,
        n_dist_joints,
        lrelu_slope=0.2,
        beta=1.0,
        n_dims=64,
        interp_mode="bilinear",
        biases=True,
    ):
        super().__init__()

        # this is the size of the output
        self.uv_size = uv_size
        self.shadow_size = shadow_size

        self.depth = 3
        self.lrelu_slope = lrelu_slope
        self.interp_mode = interp_mode
        self.align_corners = None
        if interp_mode == "bilinear":
            self.align_corners = False

        # the base number of dimensions for the shadow maps
        n_dims = n_dims

        # TODO: generate this?
        self.n_enc_dims = [
            (n_dist_joints, n_dims),
            (n_dims, n_dims),
            (n_dims, n_dims),
            (n_dims, n_dims),
        ]

        self.sizes = [shadow_size // (2**i) for i in range(len(self.n_enc_dims))]

        logger.debug(f"sizes: {self.sizes}")

        self.enc_layers = nn.ModuleList()
        for i, size in enumerate(self.sizes):
            n_in, n_out = self.n_enc_dims[i]
            logger.debug(f"EncoderLayers({i}): {n_in}, {n_out}, {size}")
            self.enc_layers.append(
                nn.Sequential(
                    la.Conv2dWNUB(
                        n_in,
                        n_out,
                        kernel_size=3,
                        height=size,
                        width=size,
                        stride=1,
                        padding=1,
                    ),
                    nn.LeakyReLU(self.lrelu_slope, inplace=True),
                )
            )

        self.n_dec_dims = [
            (n_dims, n_dims),
            (n_dims * 2, n_dims),
            (n_dims * 2, n_dims),
            (n_dims * 2, n_dims),
        ]
        self.dec_layers = nn.ModuleList()
        for i in range(len(self.sizes)):
            size = self.sizes[-i - 1]
            n_in, n_out = self.n_dec_dims[i]
            logger.debug(f"DecoderLayer({i}): {n_in}, {n_out}, {size}")

            self.dec_layers.append(
                nn.Sequential(
                    la.Conv2dWNUB(
                        n_in,
                        n_out,
                        kernel_size=3,
                        height=size,
                        width=size,
                        stride=1,
                        padding=1,
                    ),
                    nn.LeakyReLU(self.lrelu_slope, inplace=True),
                )
            )

        self.apply(weights_initializer(self.lrelu_slope))

        if biases:
            self.shadow_pred = la.Conv2dWNUB(
                self.n_dec_dims[-1][-1],
                1,
                kernel_size=3,
                height=self.sizes[0],
                width=self.sizes[0],
                stride=1,
                padding=1,
            )
        else:
            self.shadow_pred = la.Conv2dWN(
                self.n_dec_dims[-1][-1],
                1,
                kernel_size=3,
                stride=1,
                padding=1,
            )

        self.shadow_pred.apply(weights_initializer(1.0))
        self.beta = beta

    def forward(self, dist_map: th.Tensor) -> Dict[str, th.Tensor]:
        # resizing the inputs if necessary
        if dist_map.shape[-2:] != (self.shadow_size, self.shadow_size):
            dist_map = F.interpolate(dist_map, size=(self.shadow_size, self.shadow_size))

        x = dist_map

        enc_acts = []
        # unet enc
        for i, layer in enumerate(self.enc_layers):
            # TODO: try applying a 1D sparse op?
            x = layer(x)
            enc_acts.append(x)
            # TODO: add this layer elsewhere?
            if i < len(self.sizes) - 1:
                x = F.interpolate(
                    x,
                    scale_factor=0.5,
                    mode="bilinear",
                    recompute_scale_factor=True,
                    align_corners=True,
                )

        # we do not need the last one?
        for i, layer in enumerate(self.dec_layers):
            if i > 0:
                x_prev = enc_acts[-i - 1]
                x = F.interpolate(x, size=x_prev.shape[2:4], mode="bilinear", align_corners=True)
                x = th.cat([x, x_prev], dim=1)
            x = layer(x)

        shadow_map_lowres = th.sigmoid(self.shadow_pred(x) + self.beta)
        shadow_map = F.interpolate(
            shadow_map_lowres,
            (self.uv_size, self.uv_size),
            mode=self.interp_mode,
            align_corners=self.align_corners,
        )

        return {
            "shadow_map": shadow_map,
            "shadow_map_lowres": shadow_map_lowres,
        }
