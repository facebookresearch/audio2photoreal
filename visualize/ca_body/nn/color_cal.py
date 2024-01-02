"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch as th
from visualize.ca_body.utils.torch import ParamHolder

from typing import List

import torch as th
import torch.nn as nn

from torchvision.transforms.functional import gaussian_blur


logger: logging.Logger = logging.getLogger(__name__)


def scale_hook(grad: Optional[th.Tensor], scale: float) -> Optional[th.Tensor]:
    if grad is not None:
        grad = grad * scale
    return grad


class CalBase(th.nn.Module):
    def name_to_idx(self, cam_names: Sequence[str]) -> th.Tensor:
        ...


class Identity(th.nn.Module):
    def __init__(
        self,
        cameras: List[str],
        identity_camera: str,
    ) -> None:
        super().__init__()

        if identity_camera not in cameras:
            identity_camera = cameras[0]
            logger.warning(
                f"Requested color-calibration identity camera not present, defaulting to {identity_camera}."
            )

        self.identity_camera = identity_camera
        self.cameras = cameras
        self.holder = ParamHolder(
            (3 + 3,), cameras, init_value=th.tensor([1, 1, 1, 0, 0, 0], dtype=th.float32)
        )

    def name_to_idx(self, cam_names: Sequence[str]) -> th.Tensor:
        return self.holder.to_idx(cam_names)

    def forward(self, image: th.Tensor, cam_idxs: th.Tensor) -> th.Tensor:
        return image


class CalV3(CalBase):
    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, cameras, identity_camera) -> None:
        super(CalBase, self).__init__()
        # pyre-fixme[4]: Attribute must be annotated.
        self.cameras = cameras

        self.conv = th.nn.ModuleList(
            [th.nn.Conv2d(3, 3, 1, 1, 0, groups=3) for i in range(len(cameras))]
        )

        for i in range(len(cameras)):
            winit = [[1.0], [1.0], [1.0]]
            self.conv[i].weight.data[:] = th.from_numpy(
                np.array(winit, dtype=np.float32)[:, :, None, None]
            )
            self.conv[i].bias.data.zero_()

        if identity_camera not in cameras:
            identity_camera = cameras[0]
            logger.warning(
                f"Requested color-calibration identity camera not present, defaulting to {identity_camera}."
            )

        iidx = cameras.index(identity_camera)
        self.conv[iidx].weight.requires_grad = False
        self.conv[iidx].bias.requires_grad = False

    def name_to_idx(self, cam_names: Sequence[str]) -> th.Tensor:
        dev = next(self.parameters()).device
        return th.tensor([self.cameras.index(cn) for cn in cam_names], device=dev, dtype=th.long)

    def forward(self, image: th.Tensor, cam: th.Tensor) -> th.Tensor:
        return th.cat([self.conv[cam[i]](image[i : i + 1, :, :, :]) for i in range(image.size(0))])


class CalV5(CalBase):
    def __init__(
        self,
        # pyre-fixme[2]: Parameter must be annotated.
        cameras,
        # pyre-fixme[2]: Parameter must be annotated.
        identity_camera,
        gs_lrscale: float = 1e0,
        col_lrscale: float = 1e-1,
    ) -> None:
        super(CalBase, self).__init__()

        if identity_camera not in cameras:
            identity_camera = cameras[0]
            logger.warning(
                f"Requested color-calibration identity camera not present, defaulting to {identity_camera}."
            )

        # pyre-fixme[4]: Attribute must be annotated.
        self.identity_camera = identity_camera
        # pyre-fixme[4]: Attribute must be annotated.
        self.cameras = cameras
        self.gs_lrscale = gs_lrscale
        self.col_lrscale = col_lrscale
        self.holder: ParamHolder = ParamHolder(
            # pyre-fixme[6]: For 1st param expected `Tuple[int]` but got `int`.
            3 + 3,
            cameras,
            init_value=th.FloatTensor([1, 1, 1, 0, 0, 0]),
        )

        # pyre-fixme[4]: Attribute must be annotated.
        self.identity_idx = self.holder.to_idx([identity_camera]).item()
        # pyre-fixme[4]: Attribute must be annotated.
        self.grey_idxs = [self.holder.to_idx([c]).item() for c in cameras if c.startswith("41")]

        s = th.FloatTensor([0.37, 0.52, 0.52])
        self.holder.params.data[th.LongTensor(self.grey_idxs), :3] = s

    def name_to_idx(self, cam_names: Sequence[str]) -> th.Tensor:
        return self.holder.to_idx(cam_names)

    # pyre-fixme[2]: Parameter must be annotated.
    def initialize_from_texs(self, ds) -> float:
        tex_mean = ds.tex_mean.permute(1, 2, 0)
        texs = {}
        idx = 0
        while ds[idx] is None:
            idx += 1

        for cam in self.cameras:
            samp = ds[idx, cam]
            if samp is None:
                continue

            tex = samp["tex"]
            texs[cam] = tex.permute(1, 2, 0)

        stats = {}
        for cam in texs.keys():
            t = texs[cam]
            mask = (t > 0).all(dim=2)
            t = t * ds.tex_std + tex_mean
            stats[cam] = (t[mask].mean(dim=0), t[mask].std(dim=0))

        normstats = {}
        for cam in texs.keys():
            mean, std = stats[cam]
            imean, istd = stats[self.identity_camera]
            scale = istd / std
            bias = imean - scale * mean
            normstats[cam] = (scale.clamp(max=2), bias)

        for cam, nstats in normstats.items():
            cidx = self.name_to_idx([cam])[0]
            if cidx in self.grey_idxs:
                nstats = (nstats[0] / 3, nstats[1] / 3)
            self.holder.params.data[cidx, 0:3] = nstats[0]
            self.holder.params.data[cidx, 3:6] = nstats[1]
        return len(stats.keys()) / len(ds.cameras)

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    # pyre-fixme[14]: `load_state_dict` overrides method defined in `Module`
    #  inconsistently.
    def load_state_dict(self, state_dict, strict: bool = True):
        state_dict = {k[7:]: v for k, v in state_dict.items() if k.startswith("holder.")}
        return self.holder.load_state_dict(state_dict, strict=strict)

    # pyre-fixme[14]: `state_dict` overrides method defined in `Module` inconsistently.
    # pyre-fixme[3]: Return type must be annotated.
    def state_dict(
        self,
        # pyre-fixme[2]: Parameter must be annotated.
        destination=None,
        prefix: str = "",
        keep_vars: bool = False,
        saving: bool = False,
    ):
        sd = super(CalBase, self).state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        if saving:
            sd[prefix + "holder.key_list"] = self.holder.key_list
        return sd

    def forward(self, image: th.Tensor, cam_idxs: th.Tensor) -> th.Tensor:
        params = self.holder(cam_idxs)
        outs = []
        hook_scales = []
        for i in range(cam_idxs.shape[0]):
            idx = cam_idxs[i]
            img = image[i : i + 1]
            if idx == self.identity_idx:
                outs.append(img)
                hook_scales.append(1)
                continue

            w, b = params[i, :3], params[i, 3:]
            if idx in self.grey_idxs:
                b = b.sum()
                out = (img * w[None, :, None, None]).sum(dim=1, keepdim=True).expand(
                    -1, 3, -1, -1
                ) + b
            else:
                out = img * w[None, :, None, None] + b[None, :, None, None]
            outs.append(out)
            hook_scales.append(self.gs_lrscale if idx in self.grey_idxs else self.col_lrscale)

        hook_scales = th.tensor(hook_scales, device=image.device, dtype=th.float32)
        cal_out = th.cat(outs)

        if self.training and params.requires_grad:
            params.register_hook(lambda g, hs=hook_scales: scale_hook(g, hs[:, None]))
        return cal_out


class CalV6(CalBase):
    """
    A faster version of CalV5, which also does not cause CUDA synchronization. It does not support gray
    cameras.
    """

    def __init__(
        self,
        cameras: List[str],
        identity_camera: str,
    ) -> None:
        """
        Args:
            cameras (List[str]): A list of cameras.

            identity_camera (str): Name of identity camera.
        """
        super(CalBase, self).__init__()

        if identity_camera not in cameras:
            identity_camera = cameras[0]
            logger.warning(
                f"Requested color-calibration identity camera not present, defaulting to {identity_camera}."
            )

        if any(c.startswith("41") for c in cameras):
            raise ValueError("Gray cameras are not supported")

        self.identity_camera = identity_camera
        self.cameras = cameras
        self.holder = ParamHolder(
            (3 + 3,), cameras, init_value=th.as_tensor([1, 1, 1, 0, 0, 0], dtype=th.float32)
        )
        self.identity_idx: int = self.holder.key_list.index(identity_camera)
        self.register_buffer(
            "identity",
            th.as_tensor([1, 1, 1, 0, 0, 0], dtype=th.float32)[None].expand(len(cameras), -1),
            persistent=False,
        )
        identity_w = th.zeros_like(self.identity)
        identity_w[self.identity_idx, :] = 1.0
        self.register_buffer("identity_w", identity_w, persistent=False)

    # pyre-fixme[14]: `load_state_dict` overrides method defined in `Module`
    #  inconsistently.
    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True
    ) -> th.nn.modules.module._IncompatibleKeys:
        state_dict = {k[7:]: v for k, v in state_dict.items() if k.startswith("holder.")}
        return self.holder.load_state_dict(state_dict, strict=strict)

    def name_to_idx(self, cam_names: Sequence[str]) -> th.Tensor:
        dev = next(self.parameters()).device
        return th.tensor([self.cameras.index(cn) for cn in cam_names], device=dev, dtype=th.long)

    # pyre-fixme[14]: `state_dict` overrides method defined in `Module` inconsistently.
    def state_dict(
        self,
        destination: Optional[Mapping[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
        saving: bool = False,
    ) -> Mapping[str, Any]:
        sd = super(CalBase, self).state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        if saving:
            sd[prefix + "holder.key_list"] = self.holder.key_list
        return sd

    def forward(self, image: th.Tensor, cam_idxs: th.Tensor) -> th.Tensor:
        params = th.lerp(self.holder.params, self.identity, self.identity_w)[
            cam_idxs, :, None, None
        ]
        w, b = params[:, :3], params[:, 3:]
        return th.addcmul(b, w, image)


def make_cal(version: str, cal_kwargs: Dict[str, Any]) -> CalBase:
    cal_registry = {"v3": CalV3, "v5": CalV5, "v6": CalV6}

    if version not in cal_registry:
        raise ValueError(f"{version} not in {cal_registry.keys()}")

    return cal_registry[version](**cal_kwargs)

