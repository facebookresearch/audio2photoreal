"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Dict, Tuple

import numpy as np
import torch as th
import torch.nn as nn

import visualize.ca_body.nn.layers as la
from attrdict import AttrDict


class FaceDecoderFrontal(nn.Module):
    def __init__(
        self,
        assets: AttrDict,
        n_latent: int = 256,
        n_vert_out: int = 3 * 7306,
        tex_out_shp: Tuple[int, int] = (1024, 1024),
        tex_roi: Tuple[Tuple[int, int], Tuple[int, int]] = ((0, 0), (1024, 1024)),
    ) -> None:
        super().__init__()
        self.n_latent = n_latent
        self.n_vert_out = n_vert_out
        self.tex_roi = tex_roi
        self.tex_roi_shp: Tuple[int, int] = tuple(
            [int(i) for i in np.diff(np.array(tex_roi), axis=0).squeeze()]
        )
        self.tex_out_shp = tex_out_shp

        self.encmod = nn.Sequential(
            la.LinearWN(n_latent, 256), nn.LeakyReLU(0.2, inplace=True)
        )
        self.geommod = nn.Sequential(la.LinearWN(256, n_vert_out))

        self.viewmod = nn.Sequential(la.LinearWN(3, 8), nn.LeakyReLU(0.2, inplace=True))
        self.texmod2 = nn.Sequential(
            la.LinearWN(256 + 8, 256 * 4 * 4), nn.LeakyReLU(0.2, inplace=True)
        )
        self.texmod = nn.Sequential(
            la.ConvTranspose2dWNUB(256, 256, 8, 8, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            la.ConvTranspose2dWNUB(256, 128, 16, 16, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            la.ConvTranspose2dWNUB(128, 128, 32, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            la.ConvTranspose2dWNUB(128, 64, 64, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            la.ConvTranspose2dWNUB(64, 64, 128, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            la.ConvTranspose2dWNUB(64, 32, 256, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            la.ConvTranspose2dWNUB(32, 8, 512, 512, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            la.ConvTranspose2dWNUB(8, 3, 1024, 1024, 4, 2, 1),
        )

        self.bias = nn.Parameter(th.zeros(3, self.tex_roi_shp[0], self.tex_roi_shp[1]))
        self.bias.data.zero_()

        self.register_buffer(
            "frontal_view", th.as_tensor(assets.face_frontal_view, dtype=th.float32)
        )

        self.apply(lambda x: la.glorot(x, 0.2))
        la.glorot(self.texmod[-1], 1.0)

    def forward(self, face_embs: th.Tensor) -> Dict[str, th.Tensor]:
        B = face_embs.shape[0]
        view = self.frontal_view[np.newaxis].expand(B, -1)
        encout = self.encmod(face_embs)
        geomout = self.geommod(encout)
        viewout = self.viewmod(view)
        encview = th.cat([encout, viewout], dim=1)
        texout = self.texmod(self.texmod2(encview).view(-1, 256, 4, 4))
        out = {"face_geom": geomout.view(geomout.shape[0], -1, 3)}
        out["face_tex_raw"] = texout
        texout = texout + self.bias[None]
        out["face_tex"] = 255 * (texout + 0.5)
        return out
