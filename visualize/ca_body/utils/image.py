"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import warnings
from typing import Dict, Final, List, Optional, overload, Sequence, Tuple, Union

import cv2
import numpy as np
import torch as th
import torch.nn.functional as thf


Color = Tuple[np.uint8, np.uint8, np.uint8]

__DEFAULT_WB_SCALE: np.ndarray = np.array([1.05, 0.95, 1.45], dtype=np.float32)


@overload
def linear2srgb(img: th.Tensor, gamma: float = 2.4) -> th.Tensor:
    ...


@overload
def linear2srgb(img: np.ndarray, gamma: float = 2.4) -> np.ndarray:
    ...


def linear2srgb(
    img: Union[th.Tensor, np.ndarray], gamma: float = 2.4
) -> Union[th.Tensor, np.ndarray]:
    if isinstance(img, th.Tensor):
        # Note: The following combines the linear and exponential parts of the sRGB curve without
        # causing NaN values or gradients for negative inputs (where the curve would be linear).
        linear_part = img * 12.92  # linear part of sRGB curve
        exp_part = 1.055 * th.pow(th.clamp(img, min=0.0031308), 1 / gamma) - 0.055
        return th.where(img <= 0.0031308, linear_part, exp_part)
    else:
        linear_part = img * 12.92
        exp_part = 1.055 * (np.maximum(img, 0.0031308) ** (1 / gamma)) - 0.055
        return np.where(img <= 0.0031308, linear_part, exp_part)


@overload
def linear2color_corr(img: th.Tensor, dim: int = -1) -> th.Tensor:
    ...


@overload
def linear2color_corr(img: np.ndarray, dim: int = -1) -> np.ndarray:
    ...


def linear2color_corr(
    img: Union[th.Tensor, np.ndarray], dim: int = -1
) -> Union[th.Tensor, np.ndarray]:
    """Applies ad-hoc 'color correction' to a linear RGB Mugsy image along
    color channel `dim` and returns the gamma-corrected result."""

    if dim == -1:
        dim = len(img.shape) - 1

    gamma = 2.0
    black = 3.0 / 255.0
    color_scale = [1.4, 1.1, 1.6]

    assert img.shape[dim] == 3
    if dim == -1:
        dim = len(img.shape) - 1
    if isinstance(img, th.Tensor):
        scale = th.FloatTensor(color_scale).view([3 if i == dim else 1 for i in range(img.dim())])
        img = img * scale.to(img) / 1.1
        return th.clamp(
            (((1.0 / (1 - black)) * 0.95 * th.clamp(img - black, 0, 2)).pow(1.0 / gamma))
            - 15.0 / 255.0,
            0,
            2,
        )
    else:
        scale = np.array(color_scale).reshape([3 if i == dim else 1 for i in range(img.ndim)])
        img = img * scale / 1.1
        return np.clip(
            (((1.0 / (1 - black)) * 0.95 * np.clip(img - black, 0, 2)) ** (1.0 / gamma))
            - 15.0 / 255.0,
            0,
            2,
        )


def linear2displayBatch(
    val: th.Tensor,
    gamma: float = 1.5,
    wbscale: np.ndarray = __DEFAULT_WB_SCALE,
    black: float = 5.0 / 255.0,
    mode: str = "srgb",
) -> th.Tensor:
    scaling: th.Tensor = th.from_numpy(wbscale).to(val.device)
    val = val.float() / 255.0 * scaling[None, :, None, None] - black
    if mode == "srgb":
        val = linear2srgb(val, gamma=gamma)
    else:
        val = val ** th.tensor(1.0 / gamma)
    return th.clamp(val, 0, 1) * 255.0


def linear2color_corr_inv(img: th.Tensor, dim: int) -> th.Tensor:
    """Inverse of linear2color_corr.
    Removes ad-hoc 'color correction' from a gamma-corrected RGB Mugsy image
    along color channel `dim` and returns the linear RGB result."""

    gamma = 2.0
    black = 3.0 / 255.0
    color_scale = [1.4, 1.1, 1.6]

    assert img.shape[dim] == 3
    if dim == -1:
        dim = len(img.shape) - 1
    scale = th.FloatTensor(color_scale).view([3 if i == dim else 1 for i in range(img.dim())])

    img = (img + 15.0 / 255.0).pow(gamma) / (0.95 / (1 - black)) + black

    return th.clamp(img / (scale.to(img) / 1.1), 0, 1)


DEFAULT_CCM: List[List[float]] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
DEFAULT_DC_OFFSET: List[float] = [0, 0, 0]
DEFAULT_GAMMA: float = 1.0


@overload
def mapped2linear(
    img: th.Tensor,
    dim: int = -1,
    ccm: Union[List[List[float]], th.Tensor, np.ndarray] = DEFAULT_CCM,
    dc_offset: Union[List[float], th.Tensor, np.ndarray] = DEFAULT_DC_OFFSET,
    gamma: float = DEFAULT_GAMMA,
) -> th.Tensor:
    ...


@overload
def mapped2linear(
    img: np.ndarray,
    dim: int = -1,
    ccm: Union[List[List[float]], th.Tensor, np.ndarray] = DEFAULT_CCM,
    dc_offset: Union[List[float], th.Tensor, np.ndarray] = DEFAULT_DC_OFFSET,
    gamma: float = DEFAULT_GAMMA,
) -> np.ndarray:
    ...


def mapped2linear(
    img: Union[th.Tensor, np.ndarray],
    dim: int = -1,
    ccm: Union[List[List[float]], th.Tensor, np.ndarray] = DEFAULT_CCM,
    dc_offset: Union[List[float], th.Tensor, np.ndarray] = DEFAULT_DC_OFFSET,
    gamma: float = DEFAULT_GAMMA,
) -> Union[th.Tensor, np.ndarray]:
    """Maps a previously-characterized camera color space into a linear
    color space.  IMPORTANT:  This function assumes  RGB  channel order,
    not BGR.

    The characterization is specified by `ccm`, `dc_offset`, and `gamma`.
    The dimension index of the color channel is specified with `dim` (de-
    fault is -1 i.e. last dimension.)

    The function accepts both [0, 255] integer and [0, 1] float formats.
    However, the return value is always floating point in [0, 1]-range.

    FIXME(swirajaya) -
    This  is  a  reimplementation  of  `RGBMapping::map_to_lin_rgb`  in
    `//arvr/projects/codec_avatar/calibration/colorcal:colorspace`.  To
    figure out a C++ / Py binding solution that  works for both DGX and
    PROD, as well as `np.ndarray` and `th.Tensor`.

    Args:
        @param img the image in RGB, as th.Tensor or np.ndarray
        @param dim dimension of color channel
        @param ccm 3x3 color correction matrix
        @param dc_offset camera black level/dc offset
        @param gamma encoding gamma

    Returns:
        @return the corrected image as float th.Tensor or np.ndarray
    """

    assert img.shape[dim] == 3
    if dim == -1:
        dim = len(img.shape) - 1

    ndim: int = img.dim() if th.is_tensor(img) else img.ndim
    pixel_shape: List[int] = [3 if i == dim else 1 for i in range(ndim)]

    # Summation indices for CCM matrix multiplication
    # e.g. [sum_j] CCM_ij * Img_kljnpq -> ImgCorr_klinpq if say, dim == 2
    ein_ccm: List[int] = [0, 1]
    ein_inp: List[int] = [1 if i == dim else i + 2 for i in range(ndim)]
    ein_out: List[int] = [0 if i == dim else i + 2 for i in range(ndim)]

    EPS: float = 1e-7
    if isinstance(img, th.Tensor):
        if th.is_floating_point(img):
            input_saturated = img > (1.0 - EPS)
            imgf = img.double()
        else:
            input_saturated = img == 255
            imgf = img.double() / 255.0
        dc_offset = th.DoubleTensor(dc_offset).view(pixel_shape).to(img.device)
        img_linear = th.clamp(
            imgf - dc_offset,
            min=EPS,
        ).pow(1.0 / gamma)
        img_corr = th.clamp(  # CCM * img_linear
            th.einsum(th.DoubleTensor(ccm).to(img.device), ein_ccm, img_linear, ein_inp, ein_out),
            min=0.0,
            max=1.0,
        )
        img_corr = th.where(input_saturated, 1.0, img_corr)
    else:
        if np.issubdtype(img.dtype, np.floating):
            input_saturated = img > (1.0 - EPS)
            imgf = img.astype(float)
        else:
            input_saturated = img == 255
            imgf = img.astype(float) / 255.0
        dc_offset = np.array(dc_offset).reshape(pixel_shape)
        img_linear = np.clip(imgf - dc_offset, a_min=EPS, a_max=None) ** (1.0 / gamma)
        img_corr: np.ndarray = np.clip(  # CCM * img_linear
            np.einsum(np.array(ccm), ein_ccm, img_linear, ein_inp, ein_out),
            a_min=0.0,
            a_max=1.0,
        )
        img_corr: np.ndarray = np.where(input_saturated, 1.0, img_corr)

    return img_corr


@overload
def mapped2srgb(
    img: th.Tensor,
    dim: int = -1,
    ccm: Union[List[List[float]], th.Tensor, np.ndarray] = DEFAULT_CCM,
    dc_offset: Union[List[float], th.Tensor, np.ndarray] = DEFAULT_DC_OFFSET,
    gamma: float = DEFAULT_GAMMA,
) -> th.Tensor:
    ...


@overload
def mapped2srgb(
    img: np.ndarray,
    dim: int = -1,
    ccm: Union[List[List[float]], th.Tensor, np.ndarray] = DEFAULT_CCM,
    dc_offset: Union[List[float], th.Tensor, np.ndarray] = DEFAULT_DC_OFFSET,
    gamma: float = DEFAULT_GAMMA,
) -> np.ndarray:
    ...


def mapped2srgb(
    img: Union[th.Tensor, np.ndarray],
    dim: int = -1,
    ccm: Union[List[List[float]], th.Tensor, np.ndarray] = DEFAULT_CCM,
    dc_offset: Union[List[float], th.Tensor, np.ndarray] = DEFAULT_DC_OFFSET,
    gamma: float = DEFAULT_GAMMA,
) -> Union[th.Tensor, np.ndarray]:
    """Maps a previously-characterized camera color space into sRGB co-
    lor space (assuming mapped to Rec709).  IMPORTANT:  This  function
    assumes RGB channel order, not BGR.

    The characterization is specified by `ccm`, `dc_offset`, and `gamma`.
    The  dimension index  of  the color channel is specified with `dim`
    (default is -1 i.e. last dimension.)
    """
    # Note: The redundant if-statement below is due to a Pyre bug.
    # Currently Pyre fails to handle arguments into overloaded functions that are typed
    # as a union of the overloaded method parameter types.
    if isinstance(img, th.Tensor):
        return linear2srgb(mapped2linear(img, dim, ccm, dc_offset, gamma), gamma=2.4)
    else:
        return linear2srgb(mapped2linear(img, dim, ccm, dc_offset, gamma), gamma=2.4)


@overload
def srgb2linear(img: th.Tensor, gamma: float = 2.4) -> th.Tensor:
    ...


@overload
def srgb2linear(img: np.ndarray, gamma: float = 2.4) -> np.ndarray:
    ...


def srgb2linear(
    img: Union[th.Tensor, np.ndarray], gamma: float = 2.4
) -> Union[th.Tensor, np.ndarray]:
    linear_part = img / 12.92  # linear part of sRGB curve
    if isinstance(img, th.Tensor):
        # Note: The following combines the linear and exponential parts of the sRGB curve without
        # causing NaN values or gradients for negative inputs (where the curve would be linear).
        exp_part = th.pow((th.clamp(img, min=0.04045) + 0.055) / 1.055, gamma)
        return th.where(img <= 0.04045, linear_part, exp_part)
    else:
        exp_part = ((np.maximum(img, 0.04045) + 0.055) / 1.055) ** gamma
        return np.where(img <= 0.04045, linear_part, exp_part)


def scale_diff_image(diff_img: th.Tensor) -> th.Tensor:
    """Takes a difference image returns a new version scaled s.t. its values
    are remapped from [-IMG_MAX, IMG_MAX] -> [0, IMG_MAX] where IMG_MAX is
    either 1 or 255 dpeending on the range of the input."""

    mval = abs(diff_img).max().item()
    pix_range = (0, 128 if mval > 1 else 0.5, 255 if mval > 1 else 1)
    return (pix_range[1] * (diff_img / mval) + pix_range[1]).clamp(pix_range[0], pix_range[2])


class LaplacianTexture(th.nn.Module):
    def __init__(
        self, n_levels: int, n_channels: int = 3, init_scalar: Optional[float] = None
    ) -> None:
        super().__init__()
        self.n_levels = n_levels
        self.n_channels = n_channels
        if init_scalar is not None:
            init_scalar = init_scalar / n_levels

        pyr_texs = []
        for level in range(n_levels):
            if init_scalar is not None:
                pyr_texs.append(
                    th.nn.Parameter(init_scalar * th.ones(1, n_channels, 2**level, 2**level))
                )
            else:
                pyr_texs.append(th.nn.Parameter(th.zeros(1, n_channels, 2**level, 2**level)))

        self.pyr_texs = th.nn.ParameterList(pyr_texs)

    def forward(self) -> th.Tensor:
        tex = self.pyr_texs[0]
        for level in range(1, self.n_levels):
            tex = (
                thf.interpolate(tex, scale_factor=2, mode="bilinear", align_corners=False)
                + self.pyr_texs[level]
            )
        return tex

    def init_from_tex(self, tex: th.Tensor) -> None:
        ds = [tex]
        for level in range(1, self.n_levels):
            ds.append(thf.avg_pool2d(tex, 2**level))
        ds = ds[::-1]

        self.pyr_texs[0].data[:] = ds[0].data
        for level in range(1, self.n_levels):
            self.pyr_texs[level].data[:] = ds[level].data - thf.interpolate(
                ds[level - 1].data,
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            )

    def render_grad(self) -> th.Tensor:
        gtex = self.pyr_texs[0].grad
        for level in range(1, self.n_levels):
            gtex = (
                thf.interpolate(gtex, scale_factor=2, mode="bilinear", align_corners=False)
                + self.pyr_texs[level].grad
            )
        return gtex


morph_cache: Dict[Tuple[int, th.device], th.Tensor] = {}


def dilate(x: th.Tensor, ks: int) -> th.Tensor:
    assert (ks % 2) == 1
    orig_dtype = x.dtype

    if x.dtype in [th.bool, th.int64, th.int32]:
        x = x.float()
    if x.dim() == 3:
        x = x[:, None]

    if (ks, x.device) in morph_cache:
        w = morph_cache[(ks, x.device)]
    else:
        w = th.ones(1, 1, ks, ks, device=x.device)
        morph_cache[(ks, x.device)] = w

    return (thf.conv2d(x, w, padding=ks // 2) > 0).to(dtype=orig_dtype)


def erode(x: th.Tensor, ks: int) -> th.Tensor:
    if x.dtype is th.bool:
        flip_x = ~x
    else:
        flip_x = 1 - x

    flip_out = dilate(flip_x, ks)

    if flip_out.dtype is th.bool:
        return ~flip_out
    else:
        return 1 - flip_out


def smoothstep(e0: np.ndarray, e1: np.ndarray, x: np.ndarray) -> np.ndarray:
    t = np.clip(((x - e0) / (e1 - e0)), 0, 1)
    return t * t * (3.0 - 2.0 * t)


def smootherstep(e0: np.ndarray, e1: np.ndarray, x: np.ndarray) -> np.ndarray:
    t = np.clip(((x - e0) / (e1 - e0)), 0, 1)
    return (t**3) * (t * (t * 6 - 15) + 10)


def tensor2rgbjet(
    tensor: th.Tensor, x_max: Optional[float] = None, x_min: Optional[float] = None
) -> np.ndarray:
    """Converts a tensor to an uint8 image Numpy array with `cv2.COLORMAP_JET` applied.

    Args:
        tensor: Input tensor to be converted.

        x_max: The output color will be normalized as (x-x_min)/(x_max-x_min)*255.
        x_max = tensor.max() if None is given.

        x_min: The output color will be normalized as (x-x_min)/(x_max-x_min)*255.
        x_min = tensor.min() if None is given.
    """
    return cv2.applyColorMap(tensor2rgb(tensor, x_max=x_max, x_min=x_min), cv2.COLORMAP_JET)


def tensor2rgb(
    tensor: th.Tensor, x_max: Optional[float] = None, x_min: Optional[float] = None
) -> np.ndarray:
    """Converts a tensor to an uint8 image Numpy array.

    Args:
        tensor: Input tensor to be converted.

        x_max: The output color will be normalized as (x-x_min)/(x_max-x_min)*255.
        x_max = tensor.max() if None is given.

        x_min: The output color will be normalized as (x-x_min)/(x_max-x_min)*255.
        x_min = tensor.min() if None is given.
    """
    x = tensor.data.cpu().numpy()
    if x_min is None:
        x_min = x.min()
    if x_max is None:
        x_max = x.max()

    gain = 255 / np.clip(x_max - x_min, 1e-3, None)
    x = (x - x_min) * gain
    x = x.clip(0.0, 255.0)
    x = x.astype(np.uint8)
    return x


def tensor2image(
    tensor: th.Tensor,
    x_max: Optional[float] = 1.0,
    x_min: Optional[float] = 0.0,
    mode: str = "rgb",
    mask: Optional[th.Tensor] = None,
    label: Optional[str] = None,
) -> np.ndarray:
    """Converts a tensor to an image.

    Args:
        tensor: Input tensor to be converted.
        The shape of the tensor should be CxHxW or HxW. The channels are assumed to be in RGB format.

        x_max: The output color will be normalized as (x-x_min)/(x_max-x_min)*255.
        x_max = tensor.max() if None is explicitly given.

        x_min: The output color will be normalized as (x-x_min)/(x_max-x_min)*255.
        x_min = tensor.min() if None is explicitly given.

        mode: Can be `rgb` or `jet`. If `jet` is given, cv2.COLORMAP_JET would be applied.

        mask: Optional mask to be applied to the input tensor.

        label: Optional text to be added to the output image.
    """
    tensor = tensor.detach()

    # Apply mask
    if mask is not None:
        tensor = tensor * mask

    if len(tensor.size()) == 2:
        tensor = tensor[None]

    # Make three channel image
    assert len(tensor.size()) == 3, tensor.size()
    n_channels = tensor.shape[0]
    if n_channels == 1:
        tensor = tensor.repeat(3, 1, 1)
    elif n_channels != 3:
        raise ValueError(f"Unsupported number of channels {n_channels}.")

    # Convert to display format
    img = tensor.permute(1, 2, 0)

    if mode == "rgb":
        img = tensor2rgb(img, x_max=x_max, x_min=x_min)
    elif mode == "jet":
        # `cv2.applyColorMap` assumes input format in BGR
        img[:, :, :3] = img[:, :, [2, 1, 0]]
        img = tensor2rgbjet(img, x_max=x_max, x_min=x_min)
        # convert back to rgb
        img[:, :, :3] = img[:, :, [2, 1, 0]]
    else:
        raise ValueError(f"Unsupported mode {mode}.")

    if label is not None:
        img = add_label_centered(img, label)

    return img


def add_label_centered(
    img: np.ndarray,
    text: str,
    font_scale: float = 1.0,
    thickness: int = 2,
    alignment: str = "top",
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Adds label to an image

    Args:
        img: Input image.

        text: Text to be added on the image.

        font_scale: The scale of the font.

        thickness: Thinkness of the lines.

        alignment: Can be `top` or `buttom`. The alignment of the text.

        color: The color of the text. Assumes the same color space as `img`.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    textsize = cv2.getTextSize(text, font, font_scale, thickness=thickness)[0]
    img = img.astype(np.uint8).copy()

    if alignment == "top":
        cv2.putText(
            img,
            text,
            ((img.shape[1] - textsize[0]) // 2, 50),
            font,
            font_scale,
            color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
    elif alignment == "bottom":
        cv2.putText(
            img,
            text,
            ((img.shape[1] - textsize[0]) // 2, img.shape[0] - textsize[1]),
            font,
            font_scale,
            color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
    else:
        raise ValueError("Unknown text alignment")

    return img


def get_color_map(name: str = "COLORMAP_JET") -> np.ndarray:
    """Return a 256 x 3 array representing a color map from OpenCV."""
    color_map = np.arange(256, dtype=np.uint8).reshape(1, 256)
    color_map = cv2.applyColorMap(color_map, getattr(cv2, name))
    return color_map[0, :, ::-1].copy()


def feature2rgb(x: Union[th.Tensor, np.ndarray], scale: int = -1) -> np.ndarray:
    # expect 3 dim tensor
    b = (x[::3].sum(0)).data.cpu().numpy()[:, :, None]
    g = (x[1::3].sum(0)).data.cpu().numpy()[:, :, None]
    r = (x[2::3].sum(0)).data.cpu().numpy()[:, :, None]
    rgb = np.concatenate((b, g, r), axis=2)
    rgb_norm = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    rgb_norm = (rgb_norm * 255).astype(np.uint8)
    if scale != -1:
        rgb_norm = cv2.resize(rgb_norm, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return rgb_norm


def kpts2delta(kpts: th.Tensor, size: Sequence[int]) -> th.Tensor:
    # kpts: B x N x 2
    # Return: B x N x H x W x 2, 2D vectors from each grid location to kpts.
    h, w = size
    grid = th.meshgrid(
        th.arange(h, dtype=kpts.dtype, device=kpts.device),
        th.arange(w, dtype=kpts.dtype, device=kpts.device),
        indexing="xy",
    )
    delta = kpts.unflatten(-1, (1, 1, 2)) - th.stack(grid, dim=-1).unflatten(0, (1, 1, h))
    return delta


def kpts2heatmap(kpts: th.Tensor, size: Sequence[int], sigma: int = 7) -> th.Tensor:
    # kpts: B x N x 2
    dist = kpts2delta(kpts, size).square().sum(-1)
    heatmap = th.exp(-dist / (2 * sigma**2))
    return heatmap


def make_image_grid(
    data: Union[th.Tensor, Dict[str, th.Tensor]],
    keys_to_draw: Optional[List[str]] = None,
    scale_factor: Optional[float] = None,
    draw_labels: bool = True,
    grid_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Arranges a tensor of images (or a dict with labeled image tensors) into
    a grid.

    Params:
        data: Either a single image tensor [N, {1, 3}, H, W] containing images to
            arrange in a grid layout, or a dict with tensors of the same shape.
            If a dict is given, assume each entry in the dict is a batch of
            images, and form a grid where each cell contains one sample from
            each entry in the dict. Images should be in the range [0, 255].

        keys_to_draw: Select which keys in the dict should be included in each
            grid cell. If none are given, draw all keys.

        scale_factor: Optional scale factor applied to each image.

        draw_labels: Whether or not to draw the keys on each image.

        grid_size: Optionally specify the size of the resulting grid.
    """

    if isinstance(data, th.Tensor):
        data = {"": data}
        keys_to_draw = [""]

    if keys_to_draw is None:
        keys_to_draw = list(data.keys())

    n_cells = data[keys_to_draw[0]].shape[0]
    img_h = data[keys_to_draw[0]].shape[2]
    img_w = data[keys_to_draw[0]].shape[3]

    # Resize all images to match the shape of the first image, and convert
    # Greyscale -> RGB.
    for key in keys_to_draw:
        if data[key].shape[1] == 1:
            data[key] = data[key].expand(-1, 3, -1, -1)
        elif data[key].shape[1] != 3:
            raise ValueError(
                f"Image data must all be of shape [N, {1,3}, H, W]. Got shape {data[key].shape}."
            )

        data[key] = data[key].clamp(min=0, max=255)
        if data[key].shape[2] != img_h or data[key].shape[3] != img_w:
            data[key] = thf.interpolate(data[key], size=(img_h, img_w), mode="area")

        if scale_factor is not None:
            data[key] = thf.interpolate(data[key], scale_factor=scale_factor, mode="area")

    # Make an image for each grid cell by labeling and concatenating a sample
    # from each key in the data.
    cell_imgs = []
    for i in range(n_cells):
        imgs = [data[key][i].byte().cpu().numpy().transpose(1, 2, 0) for key in keys_to_draw]
        imgs = [np.ascontiguousarray(img) for img in imgs]
        if draw_labels:
            for img, label in zip(imgs, keys_to_draw):
                cv2.putText(
                    img, label, (31, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA
                )
                cv2.putText(
                    img,
                    label,
                    (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
        cell_imgs.append(np.concatenate(imgs, axis=1))

    cell_h, cell_w = cell_imgs[0].shape[:2]

    # Find the most-square grid layout.
    if grid_size is not None:
        gh, gw = grid_size
        if gh * gw < n_cells:
            raise ValueError(
                f"Requested grid size ({gh}, {gw}) (H, W) cannot hold {n_cells} images."
            )
    else:
        best_diff = np.inf
        best_side = np.inf
        best_leftover = np.inf
        gw = 0
        for gh_ in range(1, n_cells + 1):
            for gw_ in range(1, n_cells + 1):
                if gh_ * gw_ < n_cells:
                    continue

                h = gh_ * cell_h
                w = gw_ * cell_w
                diff = abs(h - w)
                max_side = max(gh_, gw_)
                leftover = gh_ * gw_ - n_cells

                if diff <= best_diff and max_side <= best_side and leftover <= best_leftover:
                    gh = gh_
                    gw = gw_
                    best_diff = diff
                    best_side = max_side
                    best_leftover = leftover

    # Put the images into the grid.
    img = np.zeros((gh * cell_h, gw * cell_w, 3), dtype=np.uint8)
    for i in range(n_cells):
        gr = i // gw
        gc = i % gw
        img[gr * cell_h : (gr + 1) * cell_h, gc * cell_w : (gc + 1) * cell_w] = cell_imgs[i]

    return img


def make_image_grid_batched(
    data: Dict[str, th.Tensor],
    max_row_hight: Optional[int] = None,
    draw_labels: bool = True,
    input_is_in_0_1: bool = False,
) -> np.ndarray:
    """A simpler version of `make_image_grid` that works for the whole batch at once.

    Usecase: A dict containing diagnostic output. All tensors in the dict have a shape of [N, {1, 3}, H, W]
    where N concides for all entries. The goal is to arranges images into a grid so that each column
    corrensponds to a key, and each row corrensponds to an index in batch.

    Example:
        Data:
            dict = {"A": A, "B": B, "C": C}

        Grid:
            | A[0] | B[0] | C[0] |
            | A[1] | B[1] | C[1] |
            | A[2] | B[2] | C[2] |

    The the grid will be aranged such way, that:
        - Each row corrensponds to an index in the batch.
        - Each column corrensponds to a key in the dict
        - For each row, images are resize such that the vertical edge matches the largest image

    Args:
        data (Dict[str, th.Tensor]): Diagnostic data.
        max_row_hight (int): The maximum allowed hight of a row.
        draw_labels (bool): Whether the keys should be drawn as labels
        input_is_in_0_1 (bool): If true, input data is assumed to be in range 0..1 otherwise in range 0..255
    """
    data_list = list(data.values())
    keys_to_draw = data.keys()

    if not all(x.ndim == 4 and (x.shape[1] == 1 or x.shape[1] == 3) for x in data_list):
        raise ValueError(
            f"Image data must all be of shape [N, {1, 3}, H, W]. Got shapes {[x.shape for x in data_list]}."
        )

    if not all(x.shape[0] == data_list[0].shape[0] for x in data_list):
        raise ValueError("Batch sizes must be the same.")

    data_list = resize_to_match(data_list, edge="vertical", max_size=max_row_hight)

    if not all(x.shape[2] == data_list[0].shape[2] for x in data_list):
        raise ValueError("Heights must be the same.")

    with th.no_grad():
        # Make all images contain 3 channels
        data_list = [x.expand(-1, 3, -1, -1) if x.shape[1] == 1 else x for x in data_list]

        # Convert to byte
        scale = 255.0 if input_is_in_0_1 else 1.0
        data_list = [x.mul(scale).round().clamp(min=0, max=255).byte() for x in data_list]

        # Convert to numpy and make it BHWC
        data_list = [x.cpu().numpy().transpose(0, 2, 3, 1) for x in data_list]

    rows = []
    # Iterate by key
    for j, label in zip(range(len(data_list)), keys_to_draw):
        col = []
        # Iterate by batch index
        for i in range(data_list[0].shape[0]):
            img = np.ascontiguousarray(data_list[j][i])
            if draw_labels:
                cv2.putText(
                    img, label, (31, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA
                )
                cv2.putText(
                    img,
                    label,
                    (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            col.append(img)
        rows.append(np.concatenate(col, axis=0))
    return np.concatenate(rows, axis=1)


def resize_to_match(
    tensors: List[th.Tensor],
    edge: str = "long",
    mode: str = "nearest",
    max_size: Optional[int] = None,
) -> List[th.Tensor]:
    """Resizes a list of image tensors s.t. a chosen edge ("long", "short", "vertical", or "horizontal")
    matches that edge on the largest image in the list."""

    assert edge in {"short", "long", "vertical", "horizontal"}
    max_shape = [max(x) for x in zip(*[t.shape for t in tensors])]

    resized_tensors = []
    for tensor in tensors:
        if edge == "long":
            edge_idx = np.argmax(tensor.shape[-2:])
        elif edge == "short":
            edge_idx = np.argmin(tensor.shape[-2:])
        elif edge == "vertical":
            edge_idx = 0
        else:  # edge == "horizontal":
            edge_idx = 1

        target_size = max_shape[-2:][edge_idx]
        if max_size is not None:
            target_size = min(max_size, max_shape[-2:][edge_idx])

        if tensor.shape[-2:][edge_idx] != target_size:
            ratio = target_size / tensor.shape[-2:][edge_idx]
            tensor = thf.interpolate(
                tensor,
                scale_factor=ratio,
                align_corners=False if mode in ["bilinear", "bicubic"] else None,
                recompute_scale_factor=True,
                mode=mode,
            )
        resized_tensors.append(tensor)
    return resized_tensors


def draw_text(
    canvas: th.Tensor,
    text: str,
    loc: Tuple[float, float],
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    scale: float = 2,
    color: Tuple[float, float, float] = (0, 0, 0),
    thickness: float = 3,
) -> th.Tensor:
    """Helper used by Rosetta to draw text on tensors using OpenCV."""
    device = canvas.device
    canvas_new = canvas.cpu().numpy().transpose(0, 2, 3, 1)
    for i in range(canvas_new.shape[0]):
        image = canvas_new[i].copy()
        if isinstance(text, list):
            cv2.putText(image, text[i], loc, font, scale, color, thickness)
        else:
            cv2.putText(image, text, loc, font, scale, color, thickness)
        canvas_new[i] = image
    canvas_tensor = th.ByteTensor(canvas_new.transpose(0, 3, 1, 2)).to(device)
    return canvas_tensor


# TODO(T153410551): Deprecate this function
def visualize_scalar_image(
    img: np.ndarray,
    min_val: float,
    val_range: float,
    color_map: int = cv2.COLORMAP_JET,
    convert_to_rgb: bool = True,
) -> np.ndarray:
    """
    Visualizes a scalar image using specified color map.
    """
    scaled_img = (img.astype(np.float32) - min_val) / val_range
    vis = cv2.applyColorMap((scaled_img * 255).clip(0, 255).astype(np.uint8), color_map)
    if convert_to_rgb:
        vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    return vis


def process_depth_image(
    depth_img: np.ndarray, depth_min: float, depth_max: float, depth_err_range: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process the depth image within the range for visualization.
    """
    valid_pixels = np.logical_and(depth_img > 0, depth_img <= depth_max)
    new_depth_img = np.zeros_like(depth_img)
    new_depth_img[valid_pixels] = depth_img[valid_pixels]
    err_image = np.abs(new_depth_img - depth_img).astype(np.float32) / depth_err_range
    return new_depth_img, err_image


def draw_keypoints(img: np.ndarray, kpt: np.ndarray, kpt_w: float) -> np.ndarray:
    """
    Draw Keypoints on given image.
    """
    x, y = kpt[:, 0], kpt[:, 1]
    w = kpt[:, 2] * kpt_w
    col = np.array([-255.0, 255.0, -255.0]) * w[:, np.newaxis]
    pts = np.column_stack((x.astype(np.int32), y.astype(np.int32)))
    for pt, c in zip(pts, col):
        cv2.circle(img, tuple(pt), 2, tuple(c), -1)

    return img


def tensor_to_rgb_array(tensor: th.Tensor) -> np.ndarray:
    """Moves channels dimension to the end of tensor.
    Makes it more suitable for visualizations.
    """
    return tensor.permute(0, 2, 3, 1).detach().cpu().numpy()


def draw_keypoints_with_color(
    image: np.ndarray, keypoints_uvw: np.ndarray, color: Color
) -> np.ndarray:
    """Renders keypoints onto a given image with particular color.
    Supports overlaps.
    """
    assert len(image.shape) == 3
    assert image.shape[-1] == 3
    coords = keypoints_uvw[:, :2].astype(np.int32)
    tmp_img = np.zeros(image.shape, dtype=np.float32)
    for uv in coords:
        cv2.circle(tmp_img, tuple(uv), 2, color, -1)
    return (image + tmp_img).clip(0.0, 255.0).astype(np.uint8)


def draw_contour(img: np.ndarray, contour_corrs: np.ndarray) -> np.ndarray:
    """
    Draw Contour on given image.
    """
    for corr in contour_corrs:
        mesh_uv = corr[1:3]
        seg_uv = corr[3:]

        x, y = int(mesh_uv[0] + 0.5), int(mesh_uv[1] + 0.5)
        cv2.circle(img, (x, y), 1, (255, 0, 0), -1)

        cv2.line(
            img,
            (int(mesh_uv[0]), int(mesh_uv[1])),
            (int(seg_uv[0]), int(seg_uv[1])),
            (-255, -255, 255),
            1,
        )

    return img
