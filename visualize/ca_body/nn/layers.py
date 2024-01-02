"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import copy
import inspect
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
import torch.nn.functional as thf
from torch.nn import init
from torch.nn.modules.utils import _pair
from torch.nn.utils.weight_norm import remove_weight_norm, WeightNorm

fc_default_activation = th.nn.LeakyReLU(0.2, inplace=True)


def gaussian_kernel(ksize: int, std: Optional[float] = None) -> np.ndarray:
    """Generates numpy array filled in with Gaussian values.

    The function generates Gaussian kernel (values according to the Gauss distribution)
    on the grid according to the kernel size.

    Args:
        ksize (int): The kernel size, must be odd number larger than 1. Otherwise throws an exception.
        std (float): The standard deviation, could be None, in which case it will be calculated
        accordoing to the kernel size.

    Returns:
        np.array: The gaussian kernel.

    """

    assert ksize % 2 == 1
    radius = ksize // 2
    if std is None:
        std = np.sqrt(-(radius**2) / (2 * np.log(0.05)))

    x, y = np.meshgrid(np.linspace(-radius, radius, ksize), np.linspace(-radius, radius, ksize))
    xy = np.stack([x, y], axis=2)
    gk = np.exp(-(xy**2).sum(-1) / (2 * std**2))
    gk /= gk.sum()
    return gk


class FCLayer(th.nn.Module):
    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, n_in, n_out, nonlin=fc_default_activation) -> None:
        super().__init__()
        self.fc = th.nn.Linear(n_in, n_out, bias=True)
        # pyre-fixme[4]: Attribute must be annotated.
        self.nonlin = nonlin if nonlin is not None else lambda x: x

        self.fc.bias.data.fill_(0)
        th.nn.init.xavier_uniform_(self.fc.weight.data)

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(self, x):
        x = self.fc(x)
        x = self.nonlin(x)
        return x


# pyre-fixme[2]: Parameter must be annotated.
def check_args_shadowing(name, method: object, arg_names) -> None:
    spec = inspect.getfullargspec(method)
    init_args = {*spec.args, *spec.kwonlyargs}
    for arg_name in arg_names:
        if arg_name in init_args:
            raise TypeError(f"{name} attempted to shadow a wrapped argument: {arg_name}")


# For backward compatibility.
class TensorMappingHook(object):
    def __init__(
        self,
        name_mapping: List[Tuple[str, str]],
        expected_shape: Optional[Dict[str, List[int]]] = None,
    ) -> None:
        """This hook is expected to be used with "_register_load_state_dict_pre_hook" to
        modify names and tensor shapes in the loaded state dictionary.

        Args:
            name_mapping: list of string tuples
            A list of tuples containing expected names from the state dict and names expected
            by the module.

            expected_shape: dict
            A mapping from parameter names to expected tensor shapes.
        """
        self.name_mapping = name_mapping
        # pyre-fixme[4]: Attribute must be annotated.
        self.expected_shape = expected_shape if expected_shape is not None else {}

    def __call__(
        self,
        # pyre-fixme[2]: Parameter must be annotated.
        state_dict,
        # pyre-fixme[2]: Parameter must be annotated.
        prefix,
        # pyre-fixme[2]: Parameter must be annotated.
        local_metadata,
        # pyre-fixme[2]: Parameter must be annotated.
        strict,
        # pyre-fixme[2]: Parameter must be annotated.
        missing_keys,
        # pyre-fixme[2]: Parameter must be annotated.
        unexpected_keys,
        # pyre-fixme[2]: Parameter must be annotated.
        error_msgs,
    ) -> None:
        for old_name, new_name in self.name_mapping:
            if prefix + old_name in state_dict:
                tensor = state_dict.pop(prefix + old_name)
                if new_name in self.expected_shape:
                    tensor = tensor.view(*self.expected_shape[new_name])
                state_dict[prefix + new_name] = tensor


# pyre-fixme[3]: Return type must be annotated.
def weight_norm_wrapper(
    cls: Type[th.nn.Module],
    new_cls_name: str,
    name: str = "weight",
    g_dim: int = 0,
    v_dim: Optional[int] = 0,
):
    """Wraps a torch.nn.Module class to support weight normalization. The wrapped class
    is compatible with the fuse/unfuse syntax and is able to load state dict from previous
    implementations.

    Args:
        cls: Type[th.nn.Module]
        Class to apply the wrapper to.

        new_cls_name: str
        Name of the new class created by the wrapper. This should be the name
        of whatever variable you assign the result of this function to. Ex:
        ``SomeLayerWN = weight_norm_wrapper(SomeLayer, "SomeLayerWN", ...)``

        name: str
        Name of the parameter to apply weight normalization to.

        g_dim: int
        Learnable dimension of the magnitude tensor. Set to None or -1 for single scalar magnitude.
        Default values for Linear and Conv2d layers are 0s and for ConvTranspose2d layers are 1s.

        v_dim: int
        Of which dimension of the direction tensor is calutated independently for the norm. Set to
        None or -1 for calculating norm over the entire direction tensor (weight tensor). Default
        values for most of the WN layers are None to preserve the existing behavior.
    """

    class Wrap(cls):
        def __init__(self, *args: Any, name=name, g_dim=g_dim, v_dim=v_dim, **kwargs: Any):
            # Check if the extra arguments are overwriting arguments for the wrapped class
            check_args_shadowing(
                "weight_norm_wrapper", super().__init__, ["name", "g_dim", "v_dim"]
            )
            super().__init__(*args, **kwargs)

            # Sanitize v_dim since we are hacking the built-in utility to support
            # a non-standard WeightNorm implementation.
            if v_dim is None:
                v_dim = -1
            self.weight_norm_args = {"name": name, "g_dim": g_dim, "v_dim": v_dim}
            self.is_fused = True
            self.unfuse()

            # For backward compatibility.
            self._register_load_state_dict_pre_hook(
                TensorMappingHook(
                    [(name, name + "_v"), ("g", name + "_g")],
                    {name + "_g": getattr(self, name + "_g").shape},
                )
            )

        def fuse(self):
            if self.is_fused:
                return
            # Check if the module is frozen.
            param_name = self.weight_norm_args["name"] + "_g"
            if hasattr(self, param_name) and param_name not in self._parameters:
                raise ValueError("Trying to fuse frozen module.")
            remove_weight_norm(self, self.weight_norm_args["name"])
            self.is_fused = True

        def unfuse(self):
            if not self.is_fused:
                return
            # Check if the module is frozen.
            param_name = self.weight_norm_args["name"]
            if hasattr(self, param_name) and param_name not in self._parameters:
                raise ValueError("Trying to unfuse frozen module.")
            wn = WeightNorm.apply(
                self, self.weight_norm_args["name"], self.weight_norm_args["g_dim"]
            )
            # Overwrite the dim property to support mismatched norm calculate for v and g tensor.
            if wn.dim != self.weight_norm_args["v_dim"]:
                wn.dim = self.weight_norm_args["v_dim"]
                # Adjust the norm values.
                weight = getattr(self, self.weight_norm_args["name"] + "_v")
                norm = getattr(self, self.weight_norm_args["name"] + "_g")
                norm.data[:] = th.norm_except_dim(weight, 2, wn.dim)
            self.is_fused = False

        def __deepcopy__(self, memo):
            # Delete derived tensor to avoid deepcopy error.
            if not self.is_fused:
                delattr(self, self.weight_norm_args["name"])

            # Deepcopy.
            cls = self.__class__
            result = cls.__new__(cls)
            memo[id(self)] = result
            for k, v in self.__dict__.items():
                setattr(result, k, copy.deepcopy(v, memo))

            if not self.is_fused:
                setattr(result, self.weight_norm_args["name"], None)
                setattr(self, self.weight_norm_args["name"], None)
            return result

    # Allows for pickling of the wrapper: https://bugs.python.org/issue13520
    Wrap.__qualname__ = new_cls_name

    return Wrap


# pyre-fixme[2]: Parameter must be annotated.
def is_weight_norm_wrapped(module) -> bool:
    for hook in module._forward_pre_hooks.values():
        if isinstance(hook, WeightNorm):
            return True
    return False


class Conv2dUB(th.nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
        # pyre-fixme[2]: Parameter must be annotated.
        *args,
        bias: bool = True,
        # pyre-fixme[2]: Parameter must be annotated.
        **kwargs,
    ) -> None:
        """Conv2d with untied bias."""
        super().__init__(in_channels, out_channels, *args, bias=False, **kwargs)
        # pyre-fixme[4]: Attribute must be annotated.
        self.bias = th.nn.Parameter(th.zeros(out_channels, height, width)) if bias else None

    # TODO: remove this method once upgraded to pytorch 1.8
    # pyre-fixme[3]: Return type must be annotated.
    def _conv_forward(self, input: th.Tensor, weight: th.Tensor, bias: Optional[th.Tensor]):
        # Copied from pt1.8 source code.
        if self.padding_mode != "zeros":
            input = thf.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            return thf.conv2d(
                input, weight, bias, self.stride, _pair(0), self.dilation, self.groups
            )
        return thf.conv2d(
            input,
            weight,
            bias,
            self.stride,
            # pyre-fixme[6]: For 5th param expected `Union[List[int], int, Size,
            #  typing.Tuple[int, ...]]` but got `Union[str, typing.Tuple[int, ...]]`.
            self.padding,
            self.dilation,
            self.groups,
        )

    def forward(self, input: th.Tensor) -> th.Tensor:
        output = self._conv_forward(input, self.weight, None)
        bias = self.bias
        if bias is not None:
            # Assertion for jit script.
            assert bias is not None
            output = output + bias[None]
        return output


class ConvTranspose2dUB(th.nn.ConvTranspose2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
        # pyre-fixme[2]: Parameter must be annotated.
        *args,
        bias: bool = True,
        # pyre-fixme[2]: Parameter must be annotated.
        **kwargs,
    ) -> None:
        """ConvTranspose2d with untied bias."""
        super().__init__(in_channels, out_channels, *args, bias=False, **kwargs)

        if self.padding_mode != "zeros":
            raise ValueError("Only `zeros` padding mode is supported for ConvTranspose2dUB")

        # pyre-fixme[4]: Attribute must be annotated.
        self.bias = th.nn.Parameter(th.zeros(out_channels, height, width)) if bias else None

    def forward(self, input: th.Tensor, output_size: Optional[List[int]] = None) -> th.Tensor:
        # TODO(T111390117): Fix Conv member annotations.
        output_padding = self._output_padding(
            input=input,
            output_size=output_size,
            # pyre-fixme[6]: For 3rd param expected `List[int]` but got
            # `Tuple[int, ...]`.
            stride=self.stride,
            # pyre-fixme[6]: For 4th param expected `List[int]` but got
            # `Union[str, typing.Tuple[int, ...]]`.
            padding=self.padding,
            # pyre-fixme[6]: For 5th param expected `List[int]` but got
            # `Tuple[int, ...]`.
            kernel_size=self.kernel_size,
            # This is now required as of D35874490
            num_spatial_dims=input.dim() - 2,
            # pyre-fixme[6]: For 6th param expected `Optional[List[int]]` but got
            # `Tuple[int, ...]`.
            dilation=self.dilation,
        )

        output = thf.conv_transpose2d(
            input,
            self.weight,
            None,
            self.stride,
            # pyre-fixme[6]: For 5th param expected `Union[List[int], int, Size,
            #  typing.Tuple[int, ...]]` but got `Union[str, typing.Tuple[int, ...]]`.
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )
        bias = self.bias
        if bias is not None:
            # Assertion for jit script.
            assert bias is not None
            output = output + bias[None]
        return output

    # NOTE: This function (on super _ConvTransposeNd) was updated in D35874490 with non-optional
    # param num_spatial_dims added. Since we need both old/new pytorch versions to work (until those
    # changes reach DGX), we're simply copying the updated code here until then.
    # TODO remove this function once updated torch code is released to DGX
    def _output_padding(
        self,
        input: th.Tensor,
        output_size: Optional[List[int]],
        stride: List[int],
        padding: List[int],
        kernel_size: List[int],
        num_spatial_dims: int,
        dilation: Optional[List[int]] = None,
    ) -> List[int]:
        if output_size is None:
            # converting to list if was not already
            ret = th.nn.modules.utils._single(self.output_padding)
        else:
            has_batch_dim = input.dim() == num_spatial_dims + 2
            num_non_spatial_dims = 2 if has_batch_dim else 1
            if len(output_size) == num_non_spatial_dims + num_spatial_dims:
                output_size = output_size[num_non_spatial_dims:]
            if len(output_size) != num_spatial_dims:
                raise ValueError(
                    "ConvTranspose{}D: for {}D input, output_size must have {} or {} elements (got {})".format(
                        num_spatial_dims,
                        input.dim(),
                        num_spatial_dims,
                        num_non_spatial_dims + num_spatial_dims,
                        len(output_size),
                    )
                )

            min_sizes = th.jit.annotate(List[int], [])
            max_sizes = th.jit.annotate(List[int], [])
            for d in range(num_spatial_dims):
                dim_size = (
                    (input.size(d + num_non_spatial_dims) - 1) * stride[d]
                    - 2 * padding[d]
                    + (dilation[d] if dilation is not None else 1) * (kernel_size[d] - 1)
                    + 1
                )
                min_sizes.append(dim_size)
                max_sizes.append(min_sizes[d] + stride[d] - 1)

            for i in range(len(output_size)):
                size = output_size[i]
                min_size = min_sizes[i]
                max_size = max_sizes[i]
                if size < min_size or size > max_size:
                    raise ValueError(
                        (
                            "requested an output size of {}, but valid sizes range "
                            "from {} to {} (for an input of {})"
                        ).format(output_size, min_sizes, max_sizes, input.size()[2:])
                    )

            res = th.jit.annotate(List[int], [])
            for d in range(num_spatial_dims):
                res.append(output_size[d] - min_sizes[d])

            ret = res
        return ret


# Set default g_dim=0 (Conv2d) or 1 (ConvTranspose2d) and v_dim=None to preserve
# the current weight norm behavior.
# pyre-fixme[5]: Global expression must be annotated.
LinearWN = weight_norm_wrapper(th.nn.Linear, "LinearWN", g_dim=0, v_dim=None)
# pyre-fixme[5]: Global expression must be annotated.
Conv2dWN = weight_norm_wrapper(th.nn.Conv2d, "Conv2dWN", g_dim=0, v_dim=None)
# pyre-fixme[5]: Global expression must be annotated.
Conv2dWNUB = weight_norm_wrapper(Conv2dUB, "Conv2dWNUB", g_dim=0, v_dim=None)
# pyre-fixme[5]: Global expression must be annotated.
ConvTranspose2dWN = weight_norm_wrapper(
    th.nn.ConvTranspose2d, "ConvTranspose2dWN", g_dim=1, v_dim=None
)
# pyre-fixme[5]: Global expression must be annotated.
ConvTranspose2dWNUB = weight_norm_wrapper(
    ConvTranspose2dUB, "ConvTranspose2dWNUB", g_dim=1, v_dim=None
)


class InterpolateHook(object):
    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, size=None, scale_factor=None, mode: str = "bilinear") -> None:
        """An object storing options for interpolate function"""
        # pyre-fixme[4]: Attribute must be annotated.
        self.size = size
        # pyre-fixme[4]: Attribute must be annotated.
        self.scale_factor = scale_factor
        self.mode = mode

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def __call__(self, module, x):
        assert len(x) == 1, "Module should take only one input for the forward method."
        return thf.interpolate(
            x[0],
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=False,
        )


# pyre-fixme[3]: Return type must be annotated.
def interpolate_wrapper(cls: Type[th.nn.Module], new_cls_name: str):
    """Wraps a torch.nn.Module class and perform additional interpolation on the
    first and only positional input of the forward method.

    Args:
        cls: Type[th.nn.Module]
        Class to apply the wrapper to.

        new_cls_name: str
        Name of the new class created by the wrapper. This should be the name
        of whatever variable you assign the result of this function to. Ex:
        ``UpConv = interpolate_wrapper(Conv, "UpConv", ...)``

    """

    class Wrap(cls):
        def __init__(
            self, *args: Any, size=None, scale_factor=None, mode="bilinear", **kwargs: Any
        ):
            check_args_shadowing(
                "interpolate_wrapper", super().__init__, ["size", "scale_factor", "mode"]
            )
            super().__init__(*args, **kwargs)
            self.register_forward_pre_hook(
                InterpolateHook(size=size, scale_factor=scale_factor, mode=mode)
            )

    # Allows for pickling of the wrapper: https://bugs.python.org/issue13520
    Wrap.__qualname__ = new_cls_name
    return Wrap


# pyre-fixme[5]: Global expression must be annotated.
UpConv2d = interpolate_wrapper(th.nn.Conv2d, "UpConv2d")
# pyre-fixme[5]: Global expression must be annotated.
UpConv2dWN = interpolate_wrapper(Conv2dWN, "UpConv2dWN")
# pyre-fixme[5]: Global expression must be annotated.
UpConv2dWNUB = interpolate_wrapper(Conv2dWNUB, "UpConv2dWNUB")


class GlobalAvgPool(th.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(self, x):
        return x.view(x.shape[0], x.shape[1], -1).mean(dim=2)


class Upsample(th.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        # pyre-fixme[4]: Attribute must be annotated.
        self.args = args
        # pyre-fixme[4]: Attribute must be annotated.
        self.kwargs = kwargs

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(self, x):
        return thf.interpolate(x, *self.args, **self.kwargs)


class DenseAffine(th.nn.Module):
    # Per-pixel affine transform layer.

    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, shape) -> None:
        super().__init__()

        self.W = th.nn.Parameter(th.ones(*shape))
        self.b = th.nn.Parameter(th.zeros(*shape))

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(self, x, scale=None, crop=None):
        W = self.W
        b = self.b

        if scale is not None:
            W = thf.interpolate(W, scale_factor=scale, mode="bilinear")
            b = thf.interpolate(b, scale_factor=scale, mode="bilinear")

        if crop is not None:
            W = W[..., crop[0] : crop[1], crop[2] : crop[3]]
            b = b[..., crop[0] : crop[1], crop[2] : crop[3]]

        return x * W + b


def glorot(m: th.nn.Module, alpha: float = 1.0) -> None:
    gain = np.sqrt(2.0 / (1.0 + alpha**2))

    if isinstance(m, th.nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, th.nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] // 4
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, th.nn.ConvTranspose3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] // 8
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, th.nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        std = gain * np.sqrt(2.0 / (n1 + n2))
    else:
        return

    is_wnw = is_weight_norm_wrapped(m)
    if is_wnw:
        m.fuse()

    m.weight.data.uniform_(-std * np.sqrt(3.0), std * np.sqrt(3.0))
    if m.bias is not None:
        m.bias.data.zero_()

    if isinstance(m, th.nn.ConvTranspose2d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]

    if is_wnw:
        m.unfuse()


def make_tuple(x: Union[int, Tuple[int, int]], n: int) -> Tuple[int, int]:
    if isinstance(x, int):
        return tuple([x for _ in range(n)])
    else:
        return x


class LinearELR(th.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gain: Optional[float] = None,
        lr_mul: float = 1.0,
        bias_lr_mul: Optional[float] = None,
    ) -> None:
        super(LinearELR, self).__init__()
        self.in_features = in_features
        self.weight = th.nn.Parameter(th.zeros(out_features, in_features, dtype=th.float32))
        if bias:
            self.bias: th.nn.Parameter = th.nn.Parameter(th.zeros(out_features, dtype=th.float32))
        else:
            self.register_parameter("bias", None)
        self.std: float = 0.0
        if gain is None:
            self.gain: float = np.sqrt(2.0)
        else:
            self.gain: float = gain
        self.lr_mul = lr_mul
        if bias_lr_mul is None:
            bias_lr_mul = lr_mul
        self.bias_lr_mul = bias_lr_mul
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.std = self.gain / np.sqrt(self.in_features) * self.lr_mul
        init.normal_(self.weight, mean=0, std=1.0 / self.lr_mul)

        if self.bias is not None:
            with th.no_grad():
                self.bias.zero_()

    def forward(self, x: th.Tensor) -> th.Tensor:
        bias = self.bias
        if bias is not None:
            bias = bias * self.bias_lr_mul
        return thf.linear(x, self.weight.mul(self.std), bias)


class Conv2dELR(th.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        output_padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        untied: bool = False,
        height: int = 1,
        width: int = 1,
        gain: Optional[float] = None,
        transpose: bool = False,
        fuse_box_filter: bool = False,
        lr_mul: float = 1.0,
        bias_lr_mul: Optional[float] = None,
    ) -> None:
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size: Tuple[int, int] = make_tuple(kernel_size, 2)
        self.stride: Tuple[int, int] = make_tuple(stride, 2)
        self.padding: Tuple[int, int] = make_tuple(padding, 2)
        self.output_padding: Tuple[int, int] = make_tuple(output_padding, 2)
        self.dilation: Tuple[int, int] = make_tuple(dilation, 2)
        self.groups = groups
        if gain is None:
            self.gain: float = np.sqrt(2.0)
        else:
            self.gain: float = gain
        self.lr_mul = lr_mul
        if bias_lr_mul is None:
            bias_lr_mul = lr_mul
        self.bias_lr_mul = bias_lr_mul
        self.transpose = transpose
        self.fan_in: float = np.prod(self.kernel_size) * in_channels // groups
        self.fuse_box_filter = fuse_box_filter
        if transpose:
            self.weight: th.nn.Parameter = th.nn.Parameter(
                th.zeros(in_channels, out_channels // groups, *self.kernel_size, dtype=th.float32)
            )
        else:
            self.weight: th.nn.Parameter = th.nn.Parameter(
                th.zeros(out_channels, in_channels // groups, *self.kernel_size, dtype=th.float32)
            )
        if bias:
            if untied:
                self.bias: th.nn.Parameter = th.nn.Parameter(
                    th.zeros(out_channels, height, width, dtype=th.float32)
                )
            else:
                self.bias: th.nn.Parameter = th.nn.Parameter(
                    th.zeros(out_channels, dtype=th.float32)
                )
        else:
            self.register_parameter("bias", None)
        self.untied = untied
        self.std: float = 0.0
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.std = self.gain / np.sqrt(self.fan_in) * self.lr_mul
        init.normal_(self.weight, mean=0, std=1.0 / self.lr_mul)

        if self.bias is not None:
            with th.no_grad():
                self.bias.zero_()

    def forward(self, x: th.Tensor) -> th.Tensor:
        if self.transpose:
            w = self.weight
            if self.fuse_box_filter:
                w = thf.pad(w, (1, 1, 1, 1), mode="constant")
                w = w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]
            bias = self.bias
            if bias is not None:
                bias = bias * self.bias_lr_mul
            out = thf.conv_transpose2d(
                x,
                w * self.std,
                bias if not self.untied else None,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            if self.untied and bias is not None:
                out = out + bias[None, ...]
            return out
        else:
            w = self.weight
            if self.fuse_box_filter:
                w = thf.pad(w, (1, 1, 1, 1), mode="constant")
                w = (
                    w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]
                ) * 0.25
            bias = self.bias
            if bias is not None:
                bias = bias * self.bias_lr_mul
            out = thf.conv2d(
                x,
                w * self.std,
                bias if not self.untied else None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            if self.untied and bias is not None:
                out = out + bias[None, ...]
            return out


class ConcatPyramid(th.nn.Module):
    def __init__(
        self,
        # pyre-fixme[2]: Parameter must be annotated.
        branch,
        # pyre-fixme[2]: Parameter must be annotated.
        n_concat_in,
        every_other: bool = True,
        ksize: int = 7,
        # pyre-fixme[2]: Parameter must be annotated.
        kstd=None,
        transposed: bool = False,
    ) -> None:
        """Module which wraps an up/down conv branch taking one input X and
        converts it into a branch which takes two inputs X, Y. At each layer of
        the original branch, we concatenate the previous output and Y,
        up/downsampling Y appropriately, before running the layer.

        Args:
            branch: th.nn.Sequential or th.nn.ModuleList
            A branch containing up/down convs, optionally separated by nonlinearities.

            n_concat_in: int
            Number of channels in the to-be-concatenated input (Y).

            every_other: bool
            If every other layer is a nonlinearity, set this flag. Default is on.

            ksize: int
            Kernel size for the Gaussian blur used to downsample each step of the pyramid.

            kstd: int
            Kernel std. dev. for the Gaussian blur used to downsample each step of the pyramid.
            If None, it is determined automatically.

            transposed: bool
            Whether or not the conv stack contains transposed convolutions or not.
        """
        super().__init__()
        assert isinstance(branch, (th.nn.Sequential, th.nn.ModuleList))

        # pyre-fixme[4]: Attribute must be annotated.
        self.branch = branch
        # pyre-fixme[4]: Attribute must be annotated.
        self.n_concat_in = n_concat_in
        self.every_other = every_other
        self.ksize = ksize
        # pyre-fixme[4]: Attribute must be annotated.
        self.kstd = kstd
        self.transposed = transposed
        if every_other:
            # pyre-fixme[4]: Attribute must be annotated.
            self.levels = int(np.ceil(len(branch) / 2))
        else:
            self.levels = len(branch)

        kernel = th.from_numpy(gaussian_kernel(ksize, kstd)).float()
        self.register_buffer("blur_kernel", kernel[None, None].expand(n_concat_in, -1, -1, -1))

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(self, x, y):
        if self.transposed:
            blurred = thf.conv2d(
                y, self.blur_kernel, groups=self.n_concat_in, padding=self.ksize // 2
            )
            pyramid = [blurred[:, :, ::2, ::2]]
        else:
            pyramid = [y]

        for _ in range(self.levels - 1):
            blurred = thf.conv2d(
                pyramid[0], self.blur_kernel, groups=self.n_concat_in, padding=self.ksize // 2
            )
            pyramid.insert(0, blurred[:, :, ::2, ::2])

        out = x
        for i, layer in enumerate(self.branch):
            if (i % 2) == 0 or not self.every_other:
                idx = i // 2 if self.every_other else i
                out = th.cat([out, pyramid[idx]], dim=1)
            out = layer(out)
        return out


# From paper "Making Convolutional Networks Shift-Invariant Again"
# https://richzhang.github.io/antialiased-cnns/
# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def get_pad_layer(pad_type):
    if pad_type in ["refl", "reflect"]:
        PadLayer = th.nn.ReflectionPad2d
    elif pad_type in ["repl", "replicate"]:
        PadLayer = th.nn.ReplicationPad2d
    elif pad_type == "zero":
        PadLayer = th.nn.ZeroPad2d
    else:
        print("Pad type [%s] not recognized" % pad_type)
    # pyre-fixme[61]: `PadLayer` is undefined, or not always defined.
    return PadLayer


class Downsample(th.nn.Module):
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, pad_type="reflect", filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample, self).__init__()
        # pyre-fixme[4]: Attribute must be annotated.
        self.filt_size = filt_size
        # pyre-fixme[4]: Attribute must be annotated.
        self.pad_off = pad_off
        # pyre-fixme[4]: Attribute must be annotated.
        self.pad_sizes = [
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
        ]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        # pyre-fixme[4]: Attribute must be annotated.
        self.stride = stride
        self.off = int((self.stride - 1) / 2.0)
        # pyre-fixme[4]: Attribute must be annotated.
        self.channels = channels

        # print('Filter size [%i]'%filt_size)
        if self.filt_size == 1:
            a = np.array(
                [
                    1.0,
                ]
            )
        elif self.filt_size == 2:
            a = np.array([1.0, 1.0])
        elif self.filt_size == 3:
            a = np.array([1.0, 2.0, 1.0])
        elif self.filt_size == 4:
            a = np.array([1.0, 3.0, 3.0, 1.0])
        elif self.filt_size == 5:
            a = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        elif self.filt_size == 6:
            a = np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
        elif self.filt_size == 7:
            a = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])

        filt = th.Tensor(a[:, None] * a[None, :])
        filt = filt / th.sum(filt)
        self.register_buffer("filt", filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        # pyre-fixme[4]: Attribute must be annotated.
        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, :: self.stride, :: self.stride]
            else:
                return self.pad(inp)[:, :, :: self.stride, :: self.stride]
        else:
            return th.nn.functional.conv2d(
                self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1]
            )
