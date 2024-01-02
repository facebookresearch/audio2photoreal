"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Any, Dict

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


def impaint_batch(value: th.Tensor, dst_ij: th.Tensor, src_ij: th.Tensor) -> th.Tensor:
    assert len(value.shape) == 4, "expecting a 4D tensor"
    preds = value[:]
    preds[:, :, dst_ij[:, 0], dst_ij[:, 1]] = value[:, :, src_ij[:, 0], src_ij[:, 1]]
    return preds


def resample_tex(tex: th.Tensor, uvs: th.Tensor, weights: th.Tensor) -> th.Tensor:
    B = tex.shape[0]
    grid = 2.0 * (uvs[np.newaxis].expand(B, -1, -1, -1) - 0.5)
    tex_resampled = F.grid_sample(tex, grid, align_corners=False, padding_mode="border")
    return (1.0 - weights) * tex + weights * tex_resampled


class SeamSampler(nn.Module):
    def __init__(self, seamless_data: Dict[str, Any]) -> None:
        super().__init__()

        self.register_buffer("dst_ij", seamless_data["dst_ij"])
        self.register_buffer("src_ij", seamless_data["src_ij"])
        self.register_buffer("uvs", seamless_data["uvs"])
        self.register_buffer("weights", seamless_data["weights"])

    def impaint(self, value: th.Tensor) -> th.Tensor:
        return impaint_batch(value, self.dst_ij, self.src_ij)

    def resample(self, tex: th.Tensor) -> th.Tensor:
        return resample_tex(tex, self.uvs, self.weights)

    def resample_border_only(self, tex: th.Tensor) -> th.Tensor:
        tex = resample_tex(tex, self.uvs, self.weights)
        return tex

    def forward(self, tex: th.Tensor) -> th.Tensor:
        x = self.impaint(tex)
        x = self.resample(x)
        return x
