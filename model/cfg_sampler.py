"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn


# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
class ClassifierFreeSampleModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  # model is the actual model to run
        self.nfeats = self.model.nfeats
        self.cond_mode = self.model.cond_mode
        self.add_frame_cond = self.model.add_frame_cond
        if self.add_frame_cond is not None:
            if self.model.resume_trans is not None:
                self.transformer = self.model.transformer
                self.tokenizer = self.model.tokenizer
            self.step = self.model.step

    def forward(self, x, timesteps, y=None):
        cond_mode = self.model.cond_mode
        assert cond_mode in ["text", "action", "audio"]
        y_uncond = deepcopy(y)
        y_uncond["uncond"] = True
        out = self.model(x, timesteps, y)
        out_uncond = self.model(x, timesteps, y_uncond)
        return out_uncond + (y["scale"].view(-1, 1, 1) * (out - out_uncond))
