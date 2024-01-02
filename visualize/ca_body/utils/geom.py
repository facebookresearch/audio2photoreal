"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.nn as nn

from sklearn.neighbors import KDTree

import logging

logger = logging.getLogger(__name__)

# NOTE: we need pytorch3d primarily for UV rasterization things
from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
from pytorch3d.structures import Meshes
from typing import Union, Optional, Tuple


def make_uv_face_index(
    vt: th.Tensor,
    vti: th.Tensor,
    uv_shape: Union[Tuple[int, int], int],
    flip_uv: bool = True,
    device: Optional[Union[str, th.device]] = None,
):
    """Compute a UV-space face index map identifying which mesh face contains each
    texel. For texels with no assigned triangle, the index will be -1."""

    if isinstance(uv_shape, int):
        uv_shape = (uv_shape, uv_shape)

    if device is not None:
        if isinstance(device, str):
            dev = th.device(device)
        else:
            dev = device
        assert dev.type == "cuda"
    else:
        dev = th.device("cuda")

    vt = 1.0 - vt.clone()

    if flip_uv:
        vt = vt.clone()
        vt[:, 1] = 1 - vt[:, 1]
    vt_pix = 2.0 * vt.to(dev) - 1.0
    vt_pix = th.cat([vt_pix, th.ones_like(vt_pix[:, 0:1])], dim=1)
    meshes = Meshes(vt_pix[np.newaxis], vti[np.newaxis].to(dev))
    with th.no_grad():
        face_index, _, _, _ = rasterize_meshes(
            meshes, uv_shape, faces_per_pixel=1, z_clip_value=0.0, bin_size=0
        )
        face_index = face_index[0, ..., 0]
    return face_index


def make_uv_vert_index(
    vt: th.Tensor,
    vi: th.Tensor,
    vti: th.Tensor,
    uv_shape: Union[Tuple[int, int], int],
    flip_uv: bool = True,
):
    """Compute a UV-space vertex index map identifying which mesh vertices
    comprise the triangle containing each texel. For texels with no assigned
    triangle, all indices will be -1.
    """
    face_index_map = make_uv_face_index(vt, vti, uv_shape, flip_uv).to(vi.device)
    vert_index_map = vi[face_index_map.clamp(min=0)]
    vert_index_map[face_index_map < 0] = -1
    return vert_index_map.long()


def bary_coords(points: th.Tensor, triangles: th.Tensor, eps: float = 1.0e-6):
    """Computes barycentric coordinates for a set of 2D query points given
    coordintes for the 3 vertices of the enclosing triangle for each point."""
    x = points[:, 0] - triangles[2, :, 0]
    x1 = triangles[0, :, 0] - triangles[2, :, 0]
    x2 = triangles[1, :, 0] - triangles[2, :, 0]
    y = points[:, 1] - triangles[2, :, 1]
    y1 = triangles[0, :, 1] - triangles[2, :, 1]
    y2 = triangles[1, :, 1] - triangles[2, :, 1]
    denom = y2 * x1 - y1 * x2
    n0 = y2 * x - x2 * y
    n1 = x1 * y - y1 * x

    # Small epsilon to prevent divide-by-zero error.
    denom = th.where(denom >= 0, denom.clamp(min=eps), denom.clamp(max=-eps))

    bary_0 = n0 / denom
    bary_1 = n1 / denom
    bary_2 = 1.0 - bary_0 - bary_1

    return th.stack((bary_0, bary_1, bary_2))


def make_uv_barys(
    vt: th.Tensor,
    vti: th.Tensor,
    uv_shape: Union[Tuple[int, int], int],
    flip_uv: bool = True,
):
    """Compute a UV-space barycentric map where each texel contains barycentric
    coordinates for that texel within its enclosing UV triangle. For texels
    with no assigned triangle, all 3 barycentric coordinates will be 0.
    """
    if isinstance(uv_shape, int):
        uv_shape = (uv_shape, uv_shape)

    if flip_uv:
        # Flip here because texture coordinates in some of our topo files are
        # stored in OpenGL convention with Y=0 on the bottom of the texture
        # unlike numpy/torch arrays/tensors.
        vt = vt.clone()
        vt[:, 1] = 1 - vt[:, 1]

    face_index_map = make_uv_face_index(vt, vti, uv_shape, flip_uv=False).to(vt.device)
    vti_map = vti.long()[face_index_map.clamp(min=0)]
    uv_tri_uvs = vt[vti_map].permute(2, 0, 1, 3)

    uv_grid = th.meshgrid(
        th.linspace(0.5, uv_shape[0] - 0.5, uv_shape[0]) / uv_shape[0],
        th.linspace(0.5, uv_shape[1] - 0.5, uv_shape[1]) / uv_shape[1],
    )
    uv_grid = th.stack(uv_grid[::-1], dim=2).to(uv_tri_uvs)

    bary_map = bary_coords(uv_grid.view(-1, 2), uv_tri_uvs.view(3, -1, 2))
    bary_map = bary_map.permute(1, 0).view(uv_shape[0], uv_shape[1], 3)
    bary_map[face_index_map < 0] = 0
    return face_index_map, bary_map


def index_image_impaint(
    index_image: th.Tensor,
    bary_image: Optional[th.Tensor] = None,
    distance_threshold=100.0,
):
    # getting the mask around the indexes?
    if len(index_image.shape) == 3:
        valid_index = (index_image != -1).any(dim=-1)
    elif len(index_image.shape) == 2:
        valid_index = index_image != -1
    else:
        raise ValueError("`index_image` should be a [H,W] or [H,W,C] image")

    invalid_index = ~valid_index

    device = index_image.device

    valid_ij = th.stack(th.where(valid_index), dim=-1)
    invalid_ij = th.stack(th.where(invalid_index), dim=-1)
    lookup_valid = KDTree(valid_ij.cpu().numpy())

    dists, idxs = lookup_valid.query(invalid_ij.cpu())

    # TODO: try average?
    idxs = th.as_tensor(idxs, device=device)[..., 0]
    dists = th.as_tensor(dists, device=device)[..., 0]

    dist_mask = dists < distance_threshold

    invalid_border = th.zeros_like(invalid_index)
    invalid_border[invalid_index] = dist_mask

    invalid_src_ij = valid_ij[idxs][dist_mask]
    invalid_dst_ij = invalid_ij[dist_mask]

    index_image_imp = index_image.clone()

    index_image_imp[invalid_dst_ij[:, 0], invalid_dst_ij[:, 1]] = index_image[
        invalid_src_ij[:, 0], invalid_src_ij[:, 1]
    ]

    if bary_image is not None:
        bary_image_imp = bary_image.clone()

        bary_image_imp[invalid_dst_ij[:, 0], invalid_dst_ij[:, 1]] = bary_image[
            invalid_src_ij[:, 0], invalid_src_ij[:, 1]
        ]

        return index_image_imp, bary_image_imp
    return index_image_imp


class GeometryModule(nn.Module):
    def __init__(
        self,
        vi,
        vt,
        vti,
        v2uv,
        uv_size,
        flip_uv=False,
        impaint=False,
        impaint_threshold=100.0,
    ):
        super().__init__()

        self.register_buffer("vi", th.as_tensor(vi))
        self.register_buffer("vt", th.as_tensor(vt))
        self.register_buffer("vti", th.as_tensor(vti))
        self.register_buffer("v2uv", th.as_tensor(v2uv, dtype=th.int64))

        # TODO: should we just pass topology here?
        self.n_verts = v2uv.shape[0]

        self.uv_size = uv_size

        # TODO: can't we just index face_index?
        index_image = make_uv_vert_index(
            self.vt, self.vi, self.vti, uv_shape=uv_size, flip_uv=flip_uv
        ).cpu()
        face_index, bary_image = make_uv_barys(
            self.vt, self.vti, uv_shape=uv_size, flip_uv=flip_uv
        )
        if impaint:
            if uv_size >= 1024:
                logger.info(
                    "impainting index image might take a while for sizes >= 1024"
                )

            index_image, bary_image = index_image_impaint(
                index_image, bary_image, impaint_threshold
            )
            # TODO: we can avoid doing this 2x
            face_index = index_image_impaint(
                face_index, distance_threshold=impaint_threshold
            )

        self.register_buffer("index_image", index_image.cpu())
        self.register_buffer("bary_image", bary_image.cpu())
        self.register_buffer("face_index_image", face_index.cpu())

    def render_index_images(self, uv_size, flip_uv=False, impaint=False):
        index_image = make_uv_vert_index(
            self.vt, self.vi, self.vti, uv_shape=uv_size, flip_uv=flip_uv
        )
        face_image, bary_image = make_uv_barys(
            self.vt, self.vti, uv_shape=uv_size, flip_uv=flip_uv
        )

        if impaint:
            index_image, bary_image = index_image_impaint(
                index_image,
                bary_image,
            )

        return index_image, face_image, bary_image

    def vn(self, verts):
        return vert_normals(verts, self.vi[np.newaxis].to(th.long))

    def to_uv(self, values):
        return values_to_uv(values, self.index_image, self.bary_image)

    def from_uv(self, values_uv):
        # TODO: we need to sample this
        return sample_uv(values_uv, self.vt, self.v2uv.to(th.long))


def sample_uv(
    values_uv,
    uv_coords,
    v2uv: Optional[th.Tensor] = None,
    mode: str = "bilinear",
    align_corners: bool = True,
    flip_uvs: bool = False,
):
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

    return values


def values_to_uv(values, index_img, bary_img):
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


def face_normals(v, vi, eps: float = 1e-5):
    pts = v[:, vi]
    v0 = pts[:, :, 1] - pts[:, :, 0]
    v1 = pts[:, :, 2] - pts[:, :, 0]
    n = th.cross(v0, v1, dim=-1)
    norm = th.norm(n, dim=-1, keepdim=True)
    norm[norm < eps] = 1
    n /= norm
    return n


def vert_normals(v, vi, eps: float = 1.0e-5):
    fnorms = face_normals(v, vi)
    fnorms = fnorms[:, :, None].expand(-1, -1, 3, -1).reshape(fnorms.shape[0], -1, 3)
    vi_flat = vi.view(1, -1).expand(v.shape[0], -1)
    vnorms = th.zeros_like(v)
    for j in range(3):
        vnorms[..., j].scatter_add_(1, vi_flat, fnorms[..., j])
    norm = th.norm(vnorms, dim=-1, keepdim=True)
    norm[norm < eps] = 1
    vnorms /= norm
    return vnorms


def compute_view_cos(verts, faces, camera_pos):
    vn = F.normalize(vert_normals(verts, faces), dim=-1)
    v2c = F.normalize(verts - camera_pos[:, np.newaxis], dim=-1)
    return th.einsum("bnd,bnd->bn", vn, v2c)


def compute_tbn(geom, vt, vi, vti):
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
    f = 1.0 / (
        vt01[None, ..., 0] * vt02[None, ..., 1]
        - vt01[None, ..., 1] * vt02[None, ..., 0]
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


def compute_v2uv(n_verts, vi, vti, n_max=4):
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
        vals = sorted(list(v2uv_dict[i]))
        v2uv[i, :] = vals[0]
        v2uv[i, : len(vals)] = np.array(vals)
    return v2uv


def compute_neighbours(n_verts, vi, n_max_values=10):
    """Computes first-ring neighbours given vertices and faces."""
    n_vi = vi.shape[0]

    adj = {i: set() for i in range(n_verts)}
    for i in range(n_vi):
        for idx in vi[i]:
            adj[idx] |= set(vi[i]) - set([idx])

    nbs_idxs = np.tile(np.arange(n_verts)[:, np.newaxis], (1, n_max_values))
    nbs_weights = np.zeros((n_verts, n_max_values), dtype=np.float32)

    for idx in range(n_verts):
        n_values = min(len(adj[idx]), n_max_values)
        nbs_idxs[idx, :n_values] = np.array(list(adj[idx]))[:n_values]
        nbs_weights[idx, :n_values] = -1.0 / n_values

    return nbs_idxs, nbs_weights


def make_postex(v, idxim, barim):
    return (
        barim[None, :, :, 0, None] * v[:, idxim[:, :, 0]]
        + barim[None, :, :, 1, None] * v[:, idxim[:, :, 1]]
        + barim[None, :, :, 2, None] * v[:, idxim[:, :, 2]]
    ).permute(0, 3, 1, 2)


def matrix_to_axisangle(r):
    th = th.arccos(0.5 * (r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2] - 1.0))[..., None]
    vec = (
        0.5
        * th.stack(
            [
                r[..., 2, 1] - r[..., 1, 2],
                r[..., 0, 2] - r[..., 2, 0],
                r[..., 1, 0] - r[..., 0, 1],
            ],
            dim=-1,
        )
        / th.sin(th)
    )
    return th, vec


def axisangle_to_matrix(rvec):
    theta = th.sqrt(1e-5 + th.sum(rvec**2, dim=-1))
    rvec = rvec / theta[..., None]
    costh = th.cos(theta)
    sinth = th.sin(theta)
    return th.stack(
        (
            th.stack(
                (
                    rvec[..., 0] ** 2 + (1.0 - rvec[..., 0] ** 2) * costh,
                    rvec[..., 0] * rvec[..., 1] * (1.0 - costh) - rvec[..., 2] * sinth,
                    rvec[..., 0] * rvec[..., 2] * (1.0 - costh) + rvec[..., 1] * sinth,
                ),
                dim=-1,
            ),
            th.stack(
                (
                    rvec[..., 0] * rvec[..., 1] * (1.0 - costh) + rvec[..., 2] * sinth,
                    rvec[..., 1] ** 2 + (1.0 - rvec[..., 1] ** 2) * costh,
                    rvec[..., 1] * rvec[..., 2] * (1.0 - costh) - rvec[..., 0] * sinth,
                ),
                dim=-1,
            ),
            th.stack(
                (
                    rvec[..., 0] * rvec[..., 2] * (1.0 - costh) - rvec[..., 1] * sinth,
                    rvec[..., 1] * rvec[..., 2] * (1.0 - costh) + rvec[..., 0] * sinth,
                    rvec[..., 2] ** 2 + (1.0 - rvec[..., 2] ** 2) * costh,
                ),
                dim=-1,
            ),
        ),
        dim=-2,
    )


def rotation_interp(r0, r1, alpha):
    r0a = r0.view(-1, 3, 3)
    r1a = r1.view(-1, 3, 3)
    r = th.bmm(r0a.permute(0, 2, 1), r1a).view_as(r0)

    th, rvec = matrix_to_axisangle(r)
    rvec = rvec * (alpha * th)

    r = axisangle_to_matrix(rvec)
    return th.bmm(r0a, r.view(-1, 3, 3)).view_as(r0)


def convert_camera_parameters(Rt, K):
    R = Rt[:, :3, :3]
    t = -R.permute(0, 2, 1).bmm(Rt[:, :3, 3].unsqueeze(2)).squeeze(2)
    return dict(
        campos=t,
        camrot=R,
        focal=K[:, :2, :2],
        princpt=K[:, :2, 2],
    )


def project_points_multi(p, Rt, K, normalize=False, size=None):
    """Project a set of 3D points into multiple cameras with a pinhole model.
    Args:
        p: [B, N, 3], input 3D points in world coordinates
        Rt: [B, NC, 3, 4], extrinsics (where NC is the number of cameras to project to)
        K: [B, NC, 3, 3], intrinsics
        normalize: bool, whether to normalize coordinates to [-1.0, 1.0]
    Returns:
        tuple:
        - [B, NC, N, 2] - projected points
        - [B, NC, N] - their
    """
    B, N = p.shape[:2]
    NC = Rt.shape[1]

    Rt = Rt.reshape(B * NC, 3, 4)
    K = K.reshape(B * NC, 3, 3)

    # [B, N, 3] -> [B * NC, N, 3]
    p = p[:, np.newaxis].expand(-1, NC, -1, -1).reshape(B * NC, -1, 3)
    p_cam = p @ Rt[:, :3, :3].mT + Rt[:, :3, 3][:, np.newaxis]
    p_pix = p_cam @ K.mT
    p_depth = p_pix[:, :, 2:]
    p_pix = (p_pix[..., :2] / p_depth).reshape(B, NC, N, 2)
    p_depth = p_depth.reshape(B, NC, N)

    if normalize:
        assert size is not None
        h, w = size
        p_pix = (
            2.0 * p_pix / th.as_tensor([w, h], dtype=th.float32, device=p.device) - 1.0
        )
    return p_pix, p_depth

def xyz2normals(xyz: th.Tensor, eps: float = 1e-8) -> th.Tensor:
    """Convert XYZ image to normal image

    Args:
        xyz: th.Tensor
        [B, 3, H, W] XYZ image

    Returns:
        th.Tensor: [B, 3, H, W] image of normals
    """

    nrml = th.zeros_like(xyz)
    xyz = th.cat((xyz[:, :, :1, :] * 0, xyz[:, :, :, :], xyz[:, :, :1, :] * 0), dim=2)
    xyz = th.cat((xyz[:, :, :, :1] * 0, xyz[:, :, :, :], xyz[:, :, :, :1] * 0), dim=3)
    U = (xyz[:, :, 2:, 1:-1] - xyz[:, :, :-2, 1:-1]) / -2
    V = (xyz[:, :, 1:-1, 2:] - xyz[:, :, 1:-1, :-2]) / -2

    nrml[:, 0, ...] = U[:, 1, ...] * V[:, 2, ...] - U[:, 2, ...] * V[:, 1, ...]
    nrml[:, 1, ...] = U[:, 2, ...] * V[:, 0, ...] - U[:, 0, ...] * V[:, 2, ...]
    nrml[:, 2, ...] = U[:, 0, ...] * V[:, 1, ...] - U[:, 1, ...] * V[:, 0, ...]
    veclen = th.norm(nrml, dim=1, keepdim=True).clamp(min=eps)
    return nrml / veclen


# pyre-fixme[2]: Parameter must be annotated.
def depth2xyz(depth, focal, princpt) -> th.Tensor:
    """Convert depth image to XYZ image using camera intrinsics

    Args:
        depth: th.Tensor
        [B, 1, H, W] depth image

        focal: th.Tensor
        [B, 2, 2] camera focal lengths

        princpt: th.Tensor
        [B, 2] camera principal points

    Returns:
        th.Tensor: [B, 3, H, W] XYZ image
    """

    b, h, w = depth.shape[0], depth.shape[2], depth.shape[3]
    ix = (
        th.arange(w, device=depth.device).float()[None, None, :] - princpt[:, None, None, 0]
    ) / focal[:, None, None, 0, 0]
    iy = (
        th.arange(h, device=depth.device).float()[None, :, None] - princpt[:, None, None, 1]
    ) / focal[:, None, None, 1, 1]
    xyz = th.zeros((b, 3, h, w), device=depth.device)
    xyz[:, 0, ...] = depth[:, 0, :, :] * ix
    xyz[:, 1, ...] = depth[:, 0, :, :] * iy
    xyz[:, 2, ...] = depth[:, 0, :, :]
    return xyz


# pyre-fixme[2]: Parameter must be annotated.
def depth2normals(depth, focal, princpt) -> th.Tensor:
    """Convert depth image to normal image using camera intrinsics

    Args:
        depth: th.Tensor
        [B, 1, H, W] depth image

        focal: th.Tensor
        [B, 2, 2] camera focal lengths

        princpt: th.Tensor
        [B, 2] camera principal points

    Returns:
        th.Tensor: [B, 3, H, W] normal image
    """

    return xyz2normals(depth2xyz(depth, focal, princpt))


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
