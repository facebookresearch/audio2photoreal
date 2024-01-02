"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
from logging import Logger

from typing import Any, Dict, Optional, Tuple, Union

import igl

import numpy as np
import torch as th

import torch.nn as nn

import torch.nn.functional as F

from visualize.ca_body.utils.geom import (
    index_image_impaint,
    make_uv_barys,
    make_uv_vert_index,
)

from trimesh import Trimesh
from trimesh.triangles import points_to_barycentric

logger: Logger = logging.getLogger(__name__)


def face_normals_v2(v: th.Tensor, vi: th.Tensor, eps: float = 1e-5) -> th.Tensor:
    pts = v[:, vi]
    v0 = pts[:, :, 1] - pts[:, :, 0]
    v1 = pts[:, :, 2] - pts[:, :, 0]
    n = th.cross(v0, v1, dim=-1)
    norm = th.norm(n, dim=-1, keepdim=True)
    norm[norm < eps] = 1
    n /= norm
    return n


def vert_normals_v2(v: th.Tensor, vi: th.Tensor, eps: float = 1.0e-5) -> th.Tensor:
    fnorms = face_normals_v2(v, vi)
    fnorms = fnorms[:, :, None].expand(-1, -1, 3, -1).reshape(fnorms.shape[0], -1, 3)
    vi_flat = vi.view(1, -1).expand(v.shape[0], -1)
    vnorms = th.zeros_like(v)
    for j in range(3):
        vnorms[..., j].scatter_add_(1, vi_flat, fnorms[..., j])
    norm = th.norm(vnorms, dim=-1, keepdim=True)
    norm[norm < eps] = 1
    vnorms /= norm
    return vnorms


def compute_neighbours(
    n_verts: int, vi: th.Tensor, n_max_values: int = 10
) -> Tuple[th.Tensor, th.Tensor]:
    """Computes first-ring neighbours given vertices and faces."""
    n_vi = vi.shape[0]

    adj = {i: set() for i in range(n_verts)}
    for i in range(n_vi):
        for idx in vi[i]:
            adj[idx] |= set(vi[i]) - {idx}

    nbs_idxs = np.tile(np.arange(n_verts)[:, np.newaxis], (1, n_max_values))
    nbs_weights = np.zeros((n_verts, n_max_values), dtype=np.float32)

    for idx in range(n_verts):
        n_values = min(len(adj[idx]), n_max_values)
        nbs_idxs[idx, :n_values] = np.array(list(adj[idx]))[:n_values]
        nbs_weights[idx, :n_values] = -1.0 / n_values

    return nbs_idxs, nbs_weights


def compute_v2uv(n_verts: int, vi: th.Tensor, vti: th.Tensor, n_max: int = 4) -> th.Tensor:
    """Computes mapping from vertex indices to texture indices.

    Args:
        vi: [F, 3], triangles
        vti: [F, 3], texture triangles
        n_max: int, max number of texture locations

    Returns:
        [n_verts, n_max], texture indices
    """
    v2uv_dict = {}
    for i_v, i_uv in zip(vi.reshape(-1), vti.reshape(-1)):
        v2uv_dict.setdefault(i_v, set()).add(i_uv)
    assert len(v2uv_dict) == n_verts
    v2uv = np.zeros((n_verts, n_max), dtype=np.int32)
    for i in range(n_verts):
        vals = sorted(v2uv_dict[i])
        v2uv[i, :] = vals[0]
        v2uv[i, : len(vals)] = np.array(vals)
    return v2uv


def values_to_uv(values: th.Tensor, index_img: th.Tensor, bary_img: th.Tensor) -> th.Tensor:
    uv_size = index_img.shape[0]
    index_mask = th.all(index_img != -1, dim=-1)
    idxs_flat = index_img[index_mask].to(th.int64)
    bary_flat = bary_img[index_mask].to(th.float32)
    # NOTE: here we assume
    values_flat = th.sum(values[:, idxs_flat].permute(0, 3, 1, 2) * bary_flat, dim=-1)
    values_uv = th.zeros(
        values.shape[0],
        values.shape[-1],
        uv_size,
        uv_size,
        dtype=values.dtype,
        device=values.device,
    )
    values_uv[:, :, index_mask] = values_flat
    return values_uv


def sample_uv(
    values_uv: th.Tensor,
    uv_coords: th.Tensor,
    v2uv: Optional[th.Tensor] = None,
    mode: str = "bilinear",
    align_corners: bool = False,
    flip_uvs: bool = False,
) -> th.Tensor:
    batch_size = values_uv.shape[0]

    if flip_uvs:
        uv_coords = uv_coords.clone()
        uv_coords[:, 1] = 1.0 - uv_coords[:, 1]

    uv_coords_norm = (uv_coords * 2.0 - 1.0)[np.newaxis, :, np.newaxis].expand(
        batch_size, -1, -1, -1
    )
    values = (
        F.grid_sample(values_uv, uv_coords_norm, align_corners=align_corners, mode=mode)
        .squeeze(-1)
        .permute((0, 2, 1))
    )

    if v2uv is not None:
        values_duplicate = values[:, v2uv]
        values = values_duplicate.mean(2)

    # if return_var:
    #     values_var = values_duplicate.var(2)
    #     return values, values_var

    return values


def compute_tbn_uv(
    tri_xyz: th.Tensor, tri_uv: th.Tensor, eps: float = 1e-5
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Compute tangents, bitangents, normals.

    Args:
        tri_xyz: [B,N,3,3] vertex coordinates
        tri_uv: [N,2] texture coordinates

    Returns:
        tangents, bitangents, normals
    """

    tri_uv = tri_uv[np.newaxis]

    v01 = tri_xyz[:, :, 1] - tri_xyz[:, :, 0]
    v02 = tri_xyz[:, :, 2] - tri_xyz[:, :, 0]

    normals = th.cross(v01, v02, dim=-1)
    normals = normals / th.norm(normals, dim=-1, keepdim=True).clamp(min=eps)

    vt01 = tri_uv[:, :, 1] - tri_uv[:, :, 0]
    vt02 = tri_uv[:, :, 2] - tri_uv[:, :, 0]

    f = th.tensor([1.0], device=tri_xyz.device) / (
        vt01[..., 0] * vt02[..., 1] - vt01[..., 1] * vt02[..., 0]
    )

    tangents = f[..., np.newaxis] * (
        v01 * vt02[..., 1][..., np.newaxis] - v02 * vt01[..., 1][..., np.newaxis]
    )
    tangents = tangents / th.norm(tangents, dim=-1, keepdim=True).clamp(min=eps)

    bitangents = th.cross(normals, tangents, dim=-1)
    bitangents = bitangents / th.norm(bitangents, dim=-1, keepdim=True).clamp(min=eps).clamp(
        min=eps
    )
    return tangents, bitangents, normals


class GeometryModule(nn.Module):
    """This module encapsulates uv correspondences and vertex images."""

    def __init__(
        self,
        vi: th.Tensor,
        vt: th.Tensor,
        vti: th.Tensor,
        v2uv: th.Tensor,
        uv_size: int,
        flip_uv: bool = False,
        impaint: bool = False,
        impaint_threshold: float = 100.0,
        device=None,
    ) -> None:
        super().__init__()

        self.register_buffer("vi", th.as_tensor(vi))
        self.register_buffer("vt", th.as_tensor(vt))
        self.register_buffer("vti", th.as_tensor(vti))
        self.register_buffer("v2uv", th.as_tensor(v2uv))

        self.uv_size: int = uv_size

        index_image = make_uv_vert_index(
            self.vt,
            self.vi,
            self.vti,
            uv_shape=uv_size,
            flip_uv=flip_uv,
        ).cpu()
        face_index, bary_image = make_uv_barys(self.vt, self.vti, uv_shape=uv_size, flip_uv=flip_uv)
        if impaint:
            # TODO: have an option to pre-compute this?
            assert isinstance(uv_size, int)
            if uv_size >= 1024:
                logger.info("impainting index image might take a while for sizes >= 1024")

            index_image, bary_image = index_image_impaint(
                index_image, bary_image, impaint_threshold
            )

        self.register_buffer("index_image", index_image.cpu())
        self.register_buffer("bary_image", bary_image.cpu())
        self.register_buffer("face_index_image", face_index.cpu())

    def render_index_images(
        self, uv_size: Union[Tuple[int, int], int], flip_uv: bool = False, impaint: bool = False
    ) -> Tuple[th.Tensor, th.Tensor]:
        index_image = make_uv_vert_index(
            self.vt, self.vi, self.vti, uv_shape=uv_size, flip_uv=flip_uv
        )
        _, bary_image = make_uv_barys(self.vt, self.vti, uv_shape=uv_size, flip_uv=flip_uv)

        if impaint:
            index_image, bary_image = index_image_impaint(
                index_image,
                bary_image,
            )

        return index_image, bary_image

    def vn(self, verts: th.Tensor) -> th.Tensor:
        return vert_normals_v2(verts, self.vi[np.newaxis].to(th.long))

    def to_uv(self, values: th.Tensor) -> th.Tensor:
        return values_to_uv(values, self.index_image, self.bary_image)

    def from_uv(self, values_uv: th.Tensor) -> th.Tensor:
        # TODO: we need to sample this
        return sample_uv(values_uv, self.vt, self.v2uv.to(th.long))


def compute_view_cos(verts: th.Tensor, faces: th.Tensor, camera_pos: th.Tensor) -> th.Tensor:
    vn = F.normalize(vert_normals_v2(verts, faces), dim=-1)
    v2c = F.normalize(verts - camera_pos[:, np.newaxis], dim=-1)
    return th.einsum("bnd,bnd->bn", vn, v2c)


def interpolate_values_mesh(
    src_values: th.Tensor, src_faces: th.Tensor, idxs: th.Tensor, weights: th.Tensor
) -> th.Tensor:
    """Interpolate values on the mesh."""
    assert src_faces.dtype == th.long, "index should be torch.long"
    assert len(src_values.shape) in [2, 3], "supporting [N, F] and [B, N, F] only"

    if src_values.shape == 2:
        return (src_values[src_faces[idxs]] * weights[..., np.newaxis]).sum(dim=1)
    else:  # src.verts.shape == 3:
        return (src_values[:, src_faces[idxs]] * weights[np.newaxis, ..., np.newaxis]).sum(dim=2)


def depth_discontuity_mask(
    depth: th.Tensor, threshold: float = 40.0, kscale: float = 4.0, pool_ksize: int = 3
) -> th.Tensor:
    device = depth.device

    with th.no_grad():
        # TODO: pass the kernel?
        kernel = th.as_tensor(
            [
                [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
                [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
            ],
            dtype=th.float32,
            device=device,
        )

        disc_mask = (th.norm(F.conv2d(depth, kernel, bias=None, padding=1), dim=1) > threshold)[
            :, np.newaxis
        ]
        disc_mask = (
            F.avg_pool2d(disc_mask.float(), pool_ksize, stride=1, padding=pool_ksize // 2) > 0.0
        )

    return disc_mask


def convert_camera_parameters(Rt: th.Tensor, K: th.Tensor) -> Dict[str, th.Tensor]:
    R = Rt[:, :3, :3]
    t = -R.permute(0, 2, 1).bmm(Rt[:, :3, 3].unsqueeze(2)).squeeze(2)
    return {
        "campos": t,
        "camrot": R,
        "focal": K[:, :2, :2],
        "princpt": K[:, :2, 2],
    }


def closest_point(mesh: Trimesh, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = mesh.vertices
    vi = mesh.faces
    # pyre-ignore
    dist, face_idxs, p = igl.point_mesh_squared_distance(points, v, vi)
    return p, dist, face_idxs


def closest_point_barycentrics(
    v: np.ndarray, vi: np.ndarray, points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Given a 3D mesh and a set of query points, return closest point barycentrics
    Args:
        v: np.array (float)
        [N, 3] mesh vertices
        vi: np.array (int)
        [N, 3] mesh triangle indices
        points: np.array (float)
        [M, 3] query points
    Returns:
        Tuple[approx, barys, interp_idxs, face_idxs]
            approx:       [M, 3] approximated (closest) points on the mesh
            barys:        [M, 3] barycentric weights that produce "approx"
            interp_idxs:  [M, 3] vertex indices for barycentric interpolation
            face_idxs:    [M] face indices for barycentric interpolation. interp_idxs = vi[face_idxs]
    """
    mesh = Trimesh(vertices=v, faces=vi)
    p, _, face_idxs = closest_point(mesh, points)
    barys = points_to_barycentric(mesh.triangles[face_idxs], p)
    b0, b1, b2 = np.split(barys, 3, axis=1)

    interp_idxs = vi[face_idxs]
    v0 = v[interp_idxs[:, 0]]
    v1 = v[interp_idxs[:, 1]]
    v2 = v[interp_idxs[:, 2]]
    approx = b0 * v0 + b1 * v1 + b2 * v2
    return approx, barys, interp_idxs, face_idxs


def make_closest_uv_barys(
    vt: np.ndarray,
    vti: np.ndarray,
    uv_shape: Union[Tuple[int, int], int],
    flip_uv: bool = True,
    return_approx_dist: bool = False,
) -> Union[Tuple[th.Tensor, th.Tensor], Tuple[th.Tensor, th.Tensor, th.Tensor]]:
    """Compute a UV-space barycentric map where each texel contains barycentric
    coordinates for the closest point on a UV triangle.
    Args:
        vt: th.Tensor
        Texture coordinates. Shape = [n_texcoords, 2]
        vti: th.Tensor
        Face texture coordinate indices. Shape = [n_faces, 3]
        uv_shape: Tuple[int, int] or int
        Shape of the texture map. (HxW)
        flip_uv: bool
        Whether or not to flip UV coordinates along the V axis (OpenGL -> numpy/pytorch convention).
        return_approx_dist: bool
        Whether or not to include the distance to the nearest point.
    Returns:
        th.Tensor: index_img: Face index image, shape [uv_shape[0], uv_shape[1]]
        th.Tensor: Barycentric coordinate map, shape [uv_shape[0], uv_shape[1], 3]
    """

    if isinstance(uv_shape, int):
        uv_shape = (uv_shape, uv_shape)

    if flip_uv:
        # Flip here because texture coordinates in some of our topo files are
        # stored in OpenGL convention with Y=0 on the bottom of the texture
        # unlike numpy/torch arrays/tensors.
        vt = vt.clone()
        vt[:, 1] = 1 - vt[:, 1]

    # Texel to UV mapping (as per OpenGL linear filtering)
    # https://www.khronos.org/registry/OpenGL/specs/gl/glspec46.core.pdf
    # Sect. 8.14, page 261
    # uv=(0.5,0.5)/w is at the center of texel [0,0]
    # uv=(w-0.5, w-0.5)/w is the center of texel [w-1,w-1]
    # texel = floor(u*w - 0.5)
    # u = (texel+0.5)/w
    uv_grid = th.meshgrid(
        th.linspace(0.5, uv_shape[0] - 1 + 0.5, uv_shape[0]) / uv_shape[0],
        th.linspace(0.5, uv_shape[1] - 1 + 0.5, uv_shape[1]) / uv_shape[1],
    )  # HxW, v,u
    uv_grid = th.stack(uv_grid[::-1], dim=2)  # HxW, u, v

    uv = uv_grid.reshape(-1, 2).data.to("cpu").numpy()
    vth = np.hstack((vt, vt[:, 0:1] * 0 + 1))
    uvh = np.hstack((uv, uv[:, 0:1] * 0 + 1))
    approx, barys, interp_idxs, face_idxs = closest_point_barycentrics(vth, vti, uvh)
    index_img = th.from_numpy(face_idxs.reshape(uv_shape[0], uv_shape[1])).long()
    bary_img = th.from_numpy(barys.reshape(uv_shape[0], uv_shape[1], 3)).float()

    if return_approx_dist:
        dist = np.linalg.norm(approx - uvh, axis=1)
        dist = th.from_numpy(dist.reshape(uv_shape[0], uv_shape[1])).float()
        return index_img, bary_img, dist
    else:
        return index_img, bary_img


def compute_tbn(
    geom: th.Tensor, vt: th.Tensor, vi: th.Tensor, vti: th.Tensor
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Computes tangent, bitangent, and normal vectors given a mesh.
    Args:
        geom: [N, n_verts, 3] th.Tensor
        Vertex positions.
        vt: [n_uv_coords, 2] th.Tensor
        UV coordinates.
        vi: [..., 3] th.Tensor
        Face vertex indices.
        vti: [..., 3] th.Tensor
        Face UV indices.
    Returns:
        [..., 3] th.Tensors for T, B, N.
    """

    v0 = geom[:, vi[..., 0]]
    v1 = geom[:, vi[..., 1]]
    v2 = geom[:, vi[..., 2]]
    vt0 = vt[vti[..., 0]]
    vt1 = vt[vti[..., 1]]
    vt2 = vt[vti[..., 2]]

    v01 = v1 - v0
    v02 = v2 - v0
    vt01 = vt1 - vt0
    vt02 = vt2 - vt0
    f = th.tensor([1.0], device=geom.device) / (
        vt01[None, ..., 0] * vt02[None, ..., 1] - vt01[None, ..., 1] * vt02[None, ..., 0]
    )
    tangent = f[..., None] * th.stack(
        [
            v01[..., 0] * vt02[None, ..., 1] - v02[..., 0] * vt01[None, ..., 1],
            v01[..., 1] * vt02[None, ..., 1] - v02[..., 1] * vt01[None, ..., 1],
            v01[..., 2] * vt02[None, ..., 1] - v02[..., 2] * vt01[None, ..., 1],
        ],
        dim=-1,
    )
    tangent = F.normalize(tangent, dim=-1)
    normal = F.normalize(th.cross(v01, v02, dim=3), dim=-1)
    bitangent = F.normalize(th.cross(tangent, normal, dim=3), dim=-1)

    return tangent, bitangent, normal


def make_postex(v: th.Tensor, idxim: th.Tensor, barim: th.Tensor) -> th.Tensor:
    return (
        barim[None, :, :, 0, None] * v[:, idxim[:, :, 0]]
        + barim[None, :, :, 1, None] * v[:, idxim[:, :, 1]]
        + barim[None, :, :, 2, None] * v[:, idxim[:, :, 2]]
    ).permute(
        0, 3, 1, 2
    )  # B x 3 x H x W


def acos_safe_th(x: th.Tensor, eps: float = 1e-4) -> th.Tensor:
    slope = th.arccos(th.as_tensor(1 - eps)) / th.as_tensor(eps)
    # TODO: stop doing this allocation once sparse gradients with NaNs (like in
    # th.where) are handled differently.
    buf = th.empty_like(x)
    good = abs(x) <= 1 - eps
    bad = ~good
    sign = th.sign(x.data[bad])
    buf[good] = th.acos(x[good])
    buf[bad] = th.acos(sign * (1 - eps)) - slope * sign * (abs(x[bad]) - 1 + eps)
    return buf


def invRodrigues(R: th.Tensor, eps: float = 1e-8) -> th.Tensor:
    """Computes the Rodrigues vectors r from the rotation matrices `R`"""

    # t = trace(R)
    # theta = rotational angle
    # [omega]_x = (R-R^T)/2
    # r = theta/sin(theta)*omega
    assert R.shape[-2:] == (3, 3)

    t = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    theta = acos_safe_th((t - 1) / 2)
    omega = (
        th.stack(
            (
                R[..., 2, 1] - R[..., 1, 2],
                R[..., 0, 2] - R[..., 2, 0],
                R[..., 1, 0] - R[..., 0, 1],
            ),
            -1,
        )
        / 2
    )

    # Edge Case 1: t >= 3 - eps
    inv_sinc = theta / th.sin(theta)
    inv_sinc_taylor_expansion = (
        1
        + (1.0 / 6.0) * th.pow(theta, 2)
        + (7.0 / 360.0) * th.pow(theta, 4)
        + (31.0 / 15120.0) * th.pow(theta, 6)
    )

    # Edge Case 2: t <= -1 + eps
    # From: https://math.stackexchange.com/questions/83874/efficient-and-accurate-numerical
    # -implementation-of-the-inverse-rodrigues-rotatio
    a = th.diagonal(R, 0, -2, -1).argmax(dim=-1)
    b = (a + 1) % 3
    c = (a + 2) % 3

    s = th.sqrt(R[..., a, a] - R[..., b, b] - R[..., c, c] + 1 + 1e-4)
    v = th.zeros_like(omega)
    v[..., a] = s / 2
    v[..., b] = (R[..., b, a] + R[..., a, b]) / (2 * s)
    v[..., c] = (R[..., c, a] + R[..., a, c]) / (2 * s)
    norm = th.norm(v, dim=-1, keepdim=True).to(v.dtype).clamp(min=eps)
    pi_vnorm = np.pi * (v / norm)

    # use taylor expansion when R is close to a identity matrix (trace(R) ~= 3)
    r = th.where(
        t[:, None] > (3 - 1e-3),
        inv_sinc_taylor_expansion[..., None] * omega,
        th.where(t[:, None] < -1 + 1e-3, pi_vnorm, inv_sinc[..., None] * omega),
    )

    return r


def EulerXYZ_to_matrix(xyz: th.Tensor) -> th.Tensor:
    # R = Rz(φ)Ry(θ)Rx(ψ) = [
    # cos θ cos φ    sin ψ sin θ cos φ − cos ψ sin φ    cos ψ sin θ cos φ + sin ψ sin φ
    # cos θ sin φ    sin ψ sin θ sin φ + cos ψ cos φ    cos ψ sin θ sin φ − sin ψ cos φ
    # − sin θ        sin ψ cos θ                        cos ψ cos θ
    # ]
    (
        x,
        y,
        z,
    ) = (
        xyz[..., 0:1],
        xyz[..., 1:2],
        xyz[..., 2:3],
    )
    sinx, cosx = th.sin(x), th.cos(x)
    siny, cosy = th.sin(y), th.cos(y)
    sinz, cosz = th.sin(z), th.cos(z)

    r1 = th.cat(
        (
            cosy * cosz,
            sinx * siny * cosz
            - cosx * sinz,  # th.sin(x) * th.sin(y) * th.cos(z) - th.cos(x) * th.sin(z),
            cosx * siny * cosz
            + sinx * sinz,  # th.cos(x) * th.sin(y) * th.cos(z) + th.sin(x) * th.sin(z)
        ),
        -1,
    )  # [..., 3]
    r3 = th.cat(
        (
            -siny,  # -th.sin(y),
            sinx * cosy,  # th.sin(x) * th.cos(y),
            cosx * cosy,  # th.cos(x) * th.cos(y)
        ),
        -1,
    )  # [..., 3]
    r2 = th.cross(r3, r1, dim=-1)

    R = th.cat((r1.unsqueeze(-2), r2.unsqueeze(-2), r3.unsqueeze(-2)), -2)
    return R


def axisangle_to_matrix(rvec: th.Tensor) -> th.Tensor:
    theta = th.sqrt(1e-5 + th.sum(th.pow(rvec, 2), dim=-1))
    rvec = rvec / theta[..., None]
    costh = th.cos(theta)
    sinth = th.sin(theta)
    return th.stack(
        (
            th.stack(
                (
                    th.pow(rvec[..., 0], 2) + (1.0 - th.pow(rvec[..., 0], 2)) * costh,
                    rvec[..., 0] * rvec[..., 1] * (1.0 - costh) - rvec[..., 2] * sinth,
                    rvec[..., 0] * rvec[..., 2] * (1.0 - costh) + rvec[..., 1] * sinth,
                ),
                dim=-1,
            ),
            th.stack(
                (
                    rvec[..., 0] * rvec[..., 1] * (1.0 - costh) + rvec[..., 2] * sinth,
                    th.pow(rvec[..., 1], 2) + (1.0 - th.pow(rvec[..., 1], 2)) * costh,
                    rvec[..., 1] * rvec[..., 2] * (1.0 - costh) - rvec[..., 0] * sinth,
                ),
                dim=-1,
            ),
            th.stack(
                (
                    rvec[..., 0] * rvec[..., 2] * (1.0 - costh) - rvec[..., 1] * sinth,
                    rvec[..., 1] * rvec[..., 2] * (1.0 - costh) + rvec[..., 0] * sinth,
                    th.pow(rvec[..., 2], 2) + (1.0 - th.pow(rvec[..., 2], 2)) * costh,
                ),
                dim=-1,
            ),
        ),
        dim=-2,
    )


def compute_view_cond_tbnrefl(
    geom: th.Tensor, campos: th.Tensor, geo_fn: GeometryModule
) -> th.Tensor:
    B = int(geom.shape[0])
    S = geo_fn.uv_size
    device = geom.device

    # TODO: this can be pre-computed, or we can assume no invalid pixels?
    mask = (geo_fn.index_image != -1).any(dim=-1)
    idxs = geo_fn.index_image[mask]
    tri_uv = geo_fn.vt[geo_fn.v2uv[idxs, 0].to(th.long)]

    tri_xyz = geom[:, idxs]

    t, b, n = compute_tbn_uv(tri_xyz, tri_uv)

    tbn_rot = th.stack((t, -b, n), dim=-2)

    tbn_rot_uv = th.zeros(
        (B, S, S, 3, 3),
        dtype=th.float32,
        device=device,
    )
    tbn_rot_uv[:, mask] = tbn_rot
    view = F.normalize(campos[:, np.newaxis] - geom, dim=-1)
    v_uv = geo_fn.to_uv(values=view)
    tbn_uv = th.einsum("bhwij,bjhw->bihw", tbn_rot_uv, v_uv)

    # reflectance vector
    n_uv = th.zeros((B, 3, S, S), dtype=th.float32, device=device)
    n_uv[..., mask] = n.permute(0, 2, 1)
    n_dot_v = (v_uv * n_uv).sum(dim=1, keepdim=True)

    r_uv = 2.0 * n_uv * n_dot_v - v_uv

    return th.cat([tbn_uv, r_uv], dim=1)


def get_barys_for_uvs(
    topology: Dict[str, Any], uv_correspondences: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a topology along with uv correspondences for the topology (eg. keypoints correspondences in uv space),
    this function will produce a tuple with the bary coordinates for each uv correspondece along with the vertex index.

    Parameters:
    ----------
    topology: Input mesh that contains vertices, faces and texture coordinates info.
    uv_correspondences: N X 2 uv locations that describe the uv correspondence to the topology

    Returns:
    -------
    bary:       (N X 3 float)
                For each uv correspondence returns the bary corrdinates for the uv pixel
    triangles: (N X 3 int)
                For each uv correspondence returns the face (i.e vertices of the faces) for that pixel.
    """
    vi: np.ndarray = topology["vi"]
    vt: np.ndarray = topology["vt"]
    vti: np.ndarray = topology["vti"]

    # # No up-down flip here
    # Here we pad the texture cordinates and correspondences with a 0
    vth = np.hstack((vt[:, :2], vt[:, :1] * 0))
    kp_uv_h = np.hstack((uv_correspondences, uv_correspondences[:, :1] * 0))

    _, kp_barys, _, face_indices = closest_point_barycentrics(vth, vti, kp_uv_h)

    kp_verts = vi[face_indices]

    return kp_barys, kp_verts
