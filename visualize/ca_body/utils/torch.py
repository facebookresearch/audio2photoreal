"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Optional, Tuple, Sequence, TypeVar, Union, Mapping, Any, List, Dict

import torch as th
import numpy as np

TensorOrContainer = Union[
    th.Tensor, str, int, Sequence["TensorOrContainer"], Mapping[str, "TensorOrContainer"]
]
NdarrayOrContainer = Union[
    np.ndarray,
    str,
    int,
    Sequence["NdarrayOrContainer"],
    Mapping[str, "NdarrayOrContainer"],
]
TensorNdarrayOrContainer = Union[
    th.Tensor,
    np.ndarray,
    str,
    int,
    Sequence["TensorNdarrayOrContainer"],
    Mapping[str, "TensorNdarrayOrContainer"],
]
TensorNdarrayModuleOrContainer = Union[
    th.Tensor,
    np.ndarray,
    th.nn.Module,
    str,
    int,
    Sequence["TensorNdarrayModuleOrContainer"],
    Mapping[str, "TensorNdarrayModuleOrContainer"],
]
TTensorOrContainer = TypeVar("TTensorOrContainer", bound=TensorOrContainer)
TNdarrayOrContainer = TypeVar("TNdarrayOrContainer", bound=NdarrayOrContainer)
TTensorNdarrayOrContainer = TypeVar("TTensorNdarrayOrContainer", bound=TensorNdarrayOrContainer)
TTensorNdarrayModuleOrContainer = TypeVar(
    "TTensorNdarrayModuleOrContainer", bound=TensorNdarrayModuleOrContainer
)


import torch as th

import logging

logger = logging.getLogger(__name__)


class ParamHolder(th.nn.Module):
    def __init__(
        self,
        param_shape: Tuple[int, ...],
        key_list: Sequence[str],
        init_value: Union[None, bool, float, int, th.Tensor] = None,
    ) -> None:
        super().__init__()

        if isinstance(param_shape, int):
            param_shape = (param_shape,)
        self.key_list: Sequence[str] = sorted(key_list)
        shp = (len(self.key_list),) + param_shape
        self.params = th.nn.Parameter(th.zeros(*shp))

        if init_value is not None:
            self.params.data[:] = init_value

    def state_dict(self, *args: Any, saving: bool = False, **kwargs: Any) -> Dict[str, Any]:
        sd = super().state_dict(*args, **kwargs)
        if saving:
            assert "key_list" not in sd
            sd["key_list"] = self.key_list
        return sd

    # pyre-fixme[14]: `load_state_dict` overrides method defined in `Module`
    #  inconsistently.
    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, **kwargs: Any
    ) -> th.nn.modules.module._IncompatibleKeys:
        # Note: Mapping is immutable while Dict is mutable. According to pyre ErrorCode[14],
        # the type of state_dict must be Mapping or supertype of Mapping to keep consistent
        # with the overrided function in its superclass.
        sd = dict(state_dict)
        if "key_list" not in sd:
            logger.warning("Missing key list list in state dict, only checking params shape.")
            assert sd["params"].shape == self.params.shape
            sd["key_list"] = self.key_list

        matching_kl = sd["key_list"] == self.key_list
        if strict:
            logger.warning("Attempting to load from mismatched key lists.")
        assert sd["params"].shape[1:] == self.params.shape[1:]

        if not matching_kl:
            src_kl = sd["key_list"]
            new_kl = sorted(set(self.key_list) | set(src_kl))
            new_shp = (len(new_kl),) + tuple(self.params.shape[1:])
            new_params = th.zeros(*new_shp, device=self.params.device)
            for f in self.key_list:
                new_params[new_kl.index(f)] = self.params[self.key_list.index(f)]
            upd = 0
            new = 0
            for f in src_kl:
                new_params[new_kl.index(f)] = sd["params"][src_kl.index(f)]
                if f in self.key_list:
                    upd += 1
                else:
                    new += 1
            logger.info(
                f"Updated {upd} keys ({100*upd/len(self.key_list):0.2f}%), added {new} new keys."
            )

            self.key_list = new_kl
            sd["params"] = new_params
            self.params = th.nn.Parameter(new_params)
        del sd["key_list"]
        return super().load_state_dict(sd, strict=strict, **kwargs)

    def to_idx(self, *args: Any) -> th.Tensor:
        if len(args) == 1:
            keys = args[0]
        else:
            keys = zip(*args)

        return th.tensor(
            [self.key_list.index(k) for k in keys],
            dtype=th.long,
            device=self.params.device,
        )

    def from_idx(self, idxs: th.Tensor) -> List[str]:
        return [self.key_list[idx] for idx in idxs]

    def forward(self, idxs: th.Tensor) -> th.Tensor:
        return self.params[idxs]
    


def to_device(
    things: TTensorNdarrayModuleOrContainer,
    device: th.device,
    cache: Optional[Dict[str, th.Tensor]] = None,
    key: Optional[str] = None,
    verbose: bool = False,
    max_bs: Optional[int] = None,
    non_blocking: bool = False,
) -> TTensorNdarrayModuleOrContainer:
    """Sends a potentially nested container of Tensors to the specified
    device. Non-tensors are preserved as-is.

    Args:
        things: Container with tensors or other containers of tensors to send
            to a GPU.

        device: Device to send the tensors to.

        cache: Optional dictionary to use as a cache for CUDAfied tensors. If
            passed, use this cache to allocate a tensor once and then resize /
            refill it on future calls to to_device() instead of reallocating
            it.

        key: If using the cache, store the tensor in this key, only for
            internal use.

        verbose: Print some info when a cached tensor is resized.

        max_bs: Maximum batch size allowed for tensors in cache

        non_blocking: if True and this copy is between CPU and GPU, the copy
            may occur asynchronously with respect to the host. For other cases,
            this argument has no effect.

    Returns:
        collection: The input collection with all tensors transferred to the given device.
    """
    device = th.device(device)

    pr = print if verbose else lambda *args, **kwargs: None

    if isinstance(things, th.Tensor) and things.device != device:
        if cache is not None:
            assert key is not None
            batch_size = things.shape[0]
            if key in cache:
                assert things.shape[1:] == cache[key].shape[1:]
                if batch_size > cache[key].shape[0]:
                    pr("Resized:", key, "from", cache[key].shape[0], "to", batch_size)
                    cache[key].resize_as_(things)
            else:
                buf_shape = list(things.shape)
                if max_bs is not None:
                    assert max_bs >= batch_size
                    buf_shape[0] = max_bs
                cache[key] = th.zeros(*buf_shape, dtype=things.dtype, device=device)
                pr("Allocated:", key, buf_shape)
            cache[key][:batch_size].copy_(things, non_blocking=non_blocking)

            return cache[key][:batch_size]
        else:
            return things.to(device, non_blocking=non_blocking)
    elif isinstance(things, th.nn.Module):
        return things.to(device, non_blocking=non_blocking)
    elif isinstance(things, dict):
        key = key + "." if key is not None else ""
        return {
            k: to_device(v, device, cache, key + k, verbose, max_bs, non_blocking)
            for k, v in things.items()
        }
    elif isinstance(things, Sequence) and not isinstance(things, str):
        key = key if key is not None else ""
        out = [
            to_device(v, device, cache, key + f"_{i}", verbose, max_bs, non_blocking)
            for i, v in enumerate(things)
        ]
        if isinstance(things, tuple):
            out = tuple(out)
        return out
    elif isinstance(things, np.ndarray):
        return to_device(th.from_numpy(things), device, cache, key, verbose, max_bs, non_blocking)
    else:
        return things



