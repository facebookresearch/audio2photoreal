"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import List

import torch as th
import torch.nn as nn

from torchvision.transforms.functional import gaussian_blur


class LearnableBlur(nn.Module):
    # TODO: should we make this conditional?
    def __init__(self, cameras: List[str]) -> None:
        super().__init__()
        self.cameras = cameras
        self.register_parameter(
            "weights_raw", nn.Parameter(th.ones(len(cameras), 3, dtype=th.float32))
        )

    def name_to_idx(self, cameras: List[str]) -> th.Tensor:
        return th.tensor(
            [self.cameras.index(c) for c in cameras],
            device=self.weights_raw.device,
            dtype=th.long,
        )

    # pyre-ignore
    def reg(self, cameras: List[str]):
        # pyre-ignore
        idxs = self.name_to_idx(cameras)
        # pyre-ignore
        return self.weights_raw[idxs]

    # pyre-ignore
    def forward(self, img: th.Tensor, cameras: List[str]):
        B = img.shape[0]
        # B, C, H, W
        idxs = self.name_to_idx(cameras)
        # TODO: mask?
        # pyre-ignore
        weights = th.softmax(self.weights_raw[idxs], dim=-1)
        weights = weights.reshape(B, 3, 1, 1, 1)
        return (
            weights[:, 0] * img
            + weights[:, 1] * gaussian_blur(img, [3, 3])
            + weights[:, 2] * gaussian_blur(img, [7, 7])
        )
