"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import List, Dict
import torch as th
import torch.nn as nn

from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer,
)

from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.textures import TexturesUV
from pytorch3d.utils import cameras_from_opencv_projection

class RenderLayer(nn.Module):
    
    def __init__(self, h, w, vi, vt, vti, flip_uvs=False):
        super().__init__()
        self.register_buffer("vi", vi, persistent=False)
        self.register_buffer("vt", vt, persistent=False)
        self.register_buffer("vti", vti, persistent=False)
        raster_settings = RasterizationSettings(image_size=(h, w))
        self.rasterizer = MeshRasterizer(raster_settings=raster_settings)
        self.flip_uvs = flip_uvs 
        image_size = th.as_tensor([h, w], dtype=th.int32)
        self.register_buffer("image_size", image_size)
    
    def forward(self, verts: th.Tensor, tex: th.Tensor, K: th.Tensor, Rt: th.Tensor, background: th.Tensor = None, output_filters: List[str] = None):

        assert output_filters is None
        assert background is None

        device = verts.device  # Get device info
        B = verts.shape[0]

        image_size = th.repeat_interleave(self.image_size[None], B, dim=0).to(device)
            
        cameras = cameras_from_opencv_projection(Rt[:,:,:3], Rt[:,:3,3], K, image_size)

        faces = self.vi[None].repeat(B, 1, 1).to(device)
        faces_uvs = self.vti[None].repeat(B, 1, 1).to(device)
        verts_uvs = self.vt[None].repeat(B, 1, 1).to(device)        
        
        # In-place operation for flipping and permuting tensor
        if not self.flip_uvs:
            tex = tex.permute(0, 2, 3, 1).flip((1,)).to(device)

        textures = TexturesUV(
            maps=tex,
            faces_uvs=faces_uvs,
            verts_uvs=verts_uvs,
        )    
        meshes = Meshes(verts.to(device), faces, textures=textures)
        
        fragments = self.rasterizer(meshes, cameras=cameras)
        rgb = meshes.sample_textures(fragments)[:,:,:,0]    
        rgb[fragments.pix_to_face[...,0] == -1] = 0.0            

        return {'render': rgb.permute(0, 3, 1, 2)}