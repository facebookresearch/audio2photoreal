"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import make_grid
from torchvision.transforms.functional import gaussian_blur

import visualize.ca_body.nn.layers as la

from visualize.ca_body.nn.blocks import (
    ConvBlock,
    ConvDownBlock,
    UpConvBlockDeep,
    tile2d,
    weights_initializer,
)
from visualize.ca_body.nn.dof_cal import LearnableBlur

from visualize.ca_body.utils.geom import (
    GeometryModule,
    compute_view_cos,
    depth_discontuity_mask,
    depth2normals,
)

from visualize.ca_body.nn.shadow import ShadowUNet, PoseToShadow
from visualize.ca_body.nn.unet import UNetWB
from visualize.ca_body.nn.color_cal import CalV5

from visualize.ca_body.utils.image import linear2displayBatch
from visualize.ca_body.utils.lbs import LBSModule
from visualize.ca_body.utils.render import RenderLayer
from visualize.ca_body.utils.seams import SeamSampler
from visualize.ca_body.utils.render import RenderLayer

from visualize.ca_body.nn.face import FaceDecoderFrontal

logger = logging.getLogger(__name__)


class CameraPixelBias(nn.Module):
    def __init__(self, image_height, image_width, cameras, ds_rate) -> None:
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.cameras = cameras
        self.n_cameras = len(cameras)

        bias = th.zeros(
            (self.n_cameras, 1, image_width // ds_rate, image_height // ds_rate), dtype=th.float32
        )
        self.register_parameter("bias", nn.Parameter(bias))

    def forward(self, idxs: th.Tensor):
        bias_up = F.interpolate(
            self.bias[idxs], size=(self.image_height, self.image_width), mode='bilinear'
        )
        return bias_up


class AutoEncoder(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        decoder_view,
        encoder_face,
        # hqlp decoder to get the codes
        decoder_face,
        shadow_net,
        upscale_net,
        assets,
        pose_to_shadow=None,
        renderer=None,
        cal=None,
        pixel_cal=None,
        learn_blur: bool = True,
    ):
        super().__init__()
        # TODO: should we have a shared LBS here?

        self.geo_fn = GeometryModule(
            assets.topology.vi,
            assets.topology.vt,
            assets.topology.vti,
            assets.topology.v2uv,
            uv_size=1024,
            impaint=True,
        )

        self.lbs_fn = LBSModule(
            assets.lbs_model_json,
            assets.lbs_config_dict,
            assets.lbs_template_verts,
            assets.lbs_scale,
            assets.global_scaling,
        )

        self.seam_sampler = SeamSampler(assets.seam_data_1024)
        self.seam_sampler_2k = SeamSampler(assets.seam_data_2048)

        # joint tex -> body and clothes
        # TODO: why do we have a joint one in the first place?
        tex_mean = gaussian_blur(th.as_tensor(assets.tex_mean)[np.newaxis], kernel_size=11)
        self.register_buffer("tex_mean", F.interpolate(tex_mean, (2048, 2048), mode='bilinear'))

        # this is shared
        self.tex_std = assets.tex_var if 'tex_var' in assets else 64.0

        face_cond_mask = th.as_tensor(assets.face_cond_mask, dtype=th.float32)[
            np.newaxis, np.newaxis
        ]
        self.register_buffer("face_cond_mask", face_cond_mask)

        meye_mask = self.geo_fn.to_uv(
            th.as_tensor(assets.mouth_eyes_mask_geom[np.newaxis, :, np.newaxis])
        )
        meye_mask = F.interpolate(meye_mask, (2048, 2048), mode='bilinear')
        self.register_buffer("meye_mask", meye_mask)

        self.decoder = ConvDecoder(
            geo_fn=self.geo_fn,
            seam_sampler=self.seam_sampler,
            **decoder,
            assets=assets,
        )

        # embs for everything but face
        non_head_mask = 1.0 - assets.face_mask
        self.encoder = Encoder(
            geo_fn=self.geo_fn,
            mask=non_head_mask,
            **encoder,
        )
        self.encoder_face = FaceEncoder(
            assets=assets,
            **encoder_face,
        )

        # using face decoder to generate better conditioning
        decoder_face_ckpt_path = None
        if 'ckpt' in decoder_face:
            decoder_face_ckpt_path = decoder_face.pop('ckpt')
        self.decoder_face = FaceDecoderFrontal(assets=assets, **decoder_face)

        if decoder_face_ckpt_path is not None:
            self.decoder_face.load_state_dict(th.load(decoder_face_ckpt_path), strict=False)

        self.decoder_view = UNetViewDecoder(
            self.geo_fn,
            seam_sampler=self.seam_sampler,
            **decoder_view,
        )

        self.shadow_net = ShadowUNet(
            ao_mean=assets.ao_mean,
            interp_mode="bilinear",
            biases=False,
            **shadow_net,
        )

        self.pose_to_shadow_enabled = False
        if pose_to_shadow is not None:
            self.pose_to_shadow_enabled = True
            self.pose_to_shadow = PoseToShadow(**pose_to_shadow)

        self.upscale_net = UpscaleNet(
            in_channels=6, size=1024, upscale_factor=2, out_channels=3, **upscale_net
        )

        self.pixel_cal_enabled = False
        if pixel_cal is not None:
            self.pixel_cal_enabled = True
            self.pixel_cal = CameraPixelBias(**pixel_cal, cameras=assets.camera_ids)

        self.learn_blur_enabled = False
        if learn_blur:
            self.learn_blur_enabled = True
            self.learn_blur = LearnableBlur(assets.camera_ids)

        # training-only stuff
        self.cal_enabled = False
        if cal is not None:
            self.cal_enabled = True
            self.cal = CalV5(**cal, cameras=assets.camera_ids)

        self.rendering_enabled = False
        if renderer is not None:
            self.rendering_enabled = True
            self.renderer = RenderLayer(
                h=renderer.image_height,
                w=renderer.image_width,
                vt=self.geo_fn.vt,
                vi=self.geo_fn.vi,
                vti=self.geo_fn.vti,
                flip_uvs=False,
            )

    @th.jit.unused
    def compute_summaries(self, preds, batch):
        # TODO: switch to common summaries?
        # return compute_summaries_mesh(preds, batch)
        rgb = linear2displayBatch(preds['rgb'][:, :3])
        rgb_gt = linear2displayBatch(batch['image'])
        depth = preds['depth'][:, np.newaxis]
        mask = depth > 0.0
        normals = (
            255 * (1.0 - depth2normals(depth, batch['focal'], batch['princpt'])) / 2.0
        ) * mask
        grid_rgb = make_grid(rgb, nrow=16).permute(1, 2, 0).clip(0, 255).to(th.uint8)
        grid_rgb_gt = make_grid(rgb_gt, nrow=16).permute(1, 2, 0).clip(0, 255).to(th.uint8)
        grid_normals = make_grid(normals, nrow=16).permute(1, 2, 0).clip(0, 255).to(th.uint8)

        progress_image = th.cat([grid_rgb, grid_rgb_gt, grid_normals], dim=0)
        return {
            'progress_image': (progress_image, 'png'),
        }

    def forward_tex(self, tex_mean_rec, tex_view_rec, shadow_map):
        x = th.cat([tex_mean_rec, tex_view_rec], dim=1)
        tex_rec = tex_mean_rec + tex_view_rec

        tex_rec = self.seam_sampler.impaint(tex_rec)
        tex_rec = self.seam_sampler.resample(tex_rec)

        tex_rec = F.interpolate(tex_rec, size=(2048, 2048), mode="bilinear", align_corners=False)
        tex_rec = tex_rec + self.upscale_net(x)

        tex_rec = tex_rec * self.tex_std + self.tex_mean

        shadow_map = self.seam_sampler_2k.impaint(shadow_map)
        shadow_map = self.seam_sampler_2k.resample(shadow_map)
        shadow_map = self.seam_sampler_2k.resample(shadow_map)

        tex_rec = tex_rec * shadow_map

        tex_rec = self.seam_sampler_2k.impaint(tex_rec)
        tex_rec = self.seam_sampler_2k.resample(tex_rec)
        tex_rec = self.seam_sampler_2k.resample(tex_rec)

        return tex_rec

    def encode(self, geom: th.Tensor, lbs_motion: th.Tensor, face_embs_hqlp: th.Tensor):

        with th.no_grad():
            verts_unposed = self.lbs_fn.unpose(geom, lbs_motion)
            verts_unposed_uv = self.geo_fn.to_uv(verts_unposed)

        # extract face region for geom + tex
        enc_preds = self.encoder(motion=lbs_motion, verts_unposed=verts_unposed)
        # TODO: probably need to rename these to `face_embs_mugsy` or smth
        # TODO: we need the same thing for face?
        # enc_face_preds = self.encoder_face(verts_unposed_uv)
        with th.no_grad():
            face_dec_preds = self.decoder_face(face_embs_hqlp)
        enc_face_preds = self.encoder_face(**face_dec_preds)

        preds = {
            **enc_preds,
            **enc_face_preds,
            'face_dec_preds': face_dec_preds,
        }
        return preds

    def forward(
        self,
        # TODO: should we try using this as well for cond?
        lbs_motion: th.Tensor,
        campos: th.Tensor,
        geom: Optional[th.Tensor] = None,
        ao: Optional[th.Tensor] = None,
        K: Optional[th.Tensor] = None,
        Rt: Optional[th.Tensor] = None,
        image_bg: Optional[th.Tensor] = None,
        image: Optional[th.Tensor] = None,
        image_mask: Optional[th.Tensor] = None,
        embs: Optional[th.Tensor] = None,
        _index: Optional[Dict[str, th.Tensor]] = None,
        face_embs: Optional[th.Tensor] = None,
        embs_conv: Optional[th.Tensor] = None,
        tex_seg: Optional[th.Tensor] = None,
        encode=True,
        iteration: Optional[int] = None,
        **kwargs,
    ):
        B = lbs_motion.shape[0]

        if not th.jit.is_scripting() and encode:
            # NOTE: these are `face_embs_hqlp`
            enc_preds = self.encode(geom, lbs_motion, face_embs)
            embs = enc_preds['embs']
            # NOTE: these are `face_embs` in body space
            face_embs_body = enc_preds['face_embs']

        dec_preds = self.decoder(
            motion=lbs_motion,
            embs=embs,
            face_embs=face_embs_body,
            embs_conv=embs_conv,
        )

        geom_rec = self.lbs_fn.pose(dec_preds['geom_delta_rec'], lbs_motion)

        dec_view_preds = self.decoder_view(
            geom_rec=geom_rec,
            tex_mean_rec=dec_preds["tex_mean_rec"],
            camera_pos=campos,
        )

        # TODO: should we train an AO model?
        if self.training and self.pose_to_shadow_enabled:
            shadow_preds = self.shadow_net(ao_map=ao)
            pose_shadow_preds = self.pose_to_shadow(lbs_motion)
            shadow_preds['pose_shadow_map'] = pose_shadow_preds['shadow_map']
        elif self.pose_to_shadow_enabled:
            shadow_preds = self.pose_to_shadow(lbs_motion)
        else:
            shadow_preds = self.shadow_net(ao_map=ao)

        tex_rec = self.forward_tex(
            dec_preds["tex_mean_rec"],
            dec_view_preds["tex_view_rec"],
            shadow_preds["shadow_map"],
        )

        if not th.jit.is_scripting() and self.cal_enabled:
            tex_rec = self.cal(tex_rec, self.cal.name_to_idx(_index['camera']))

        preds = {
            'geom': geom_rec,
            'tex_rec': tex_rec,
            **dec_preds,
            **shadow_preds,
            **dec_view_preds,
        }

        if not th.jit.is_scripting() and encode:
            preds.update(**enc_preds)

        if not th.jit.is_scripting() and self.rendering_enabled:

            # NOTE: this is a reduced version tested for forward only
            renders = self.renderer(
                preds['geom'],
                tex_rec,
                K=K,
                Rt=Rt,
            )

            preds.update(rgb=renders['render'])

        if not th.jit.is_scripting() and self.learn_blur_enabled:
            preds['rgb'] = self.learn_blur(preds['rgb'], _index['camera'])
            preds['learn_blur_weights'] = self.learn_blur.reg(_index['camera'])

        if not th.jit.is_scripting() and self.pixel_cal_enabled:
            assert self.cal_enabled
            cam_idxs = self.cal.name_to_idx(_index['camera'])
            pixel_bias = self.pixel_cal(cam_idxs)
            preds['rgb'] = preds['rgb'] + pixel_bias

        return preds


class Encoder(nn.Module):
    """A joint encoder for tex and geometry."""

    def __init__(
        self,
        geo_fn,
        n_embs,
        noise_std,
        mask,
        logvar_scale=0.1,
    ):
        """Fixed-width conv encoder."""
        super().__init__()

        self.noise_std = noise_std
        self.n_embs = n_embs
        self.geo_fn = geo_fn
        self.logvar_scale = logvar_scale

        self.verts_conv = ConvDownBlock(3, 8, 512)

        mask = th.as_tensor(mask[np.newaxis, np.newaxis], dtype=th.float32)
        mask = F.interpolate(mask, size=(512, 512), mode='bilinear').to(th.bool)
        self.register_buffer("mask", mask)

        self.joint_conv_blocks = nn.Sequential(
            ConvDownBlock(8, 16, 256),
            ConvDownBlock(16, 32, 128),
            ConvDownBlock(32, 32, 64),
            ConvDownBlock(32, 64, 32),
            ConvDownBlock(64, 128, 16),
            ConvDownBlock(128, 128, 8),
            # ConvDownBlock(128, 128, 4),
        )

        # TODO: should we put initializer
        self.mu = la.LinearWN(4 * 4 * 128, self.n_embs)
        self.logvar = la.LinearWN(4 * 4 * 128, self.n_embs)

        self.apply(weights_initializer(0.2))
        self.mu.apply(weights_initializer(1.0))
        self.logvar.apply(weights_initializer(1.0))

    def forward(self, motion, verts_unposed):
        preds = {}

        B = motion.shape[0]

        # converting motion to the unposed
        verts_cond = (
            F.interpolate(self.geo_fn.to_uv(verts_unposed), size=(512, 512), mode='bilinear')
            * self.mask
        )
        verts_cond = self.verts_conv(verts_cond)

        # tex_cond = F.interpolate(tex_avg, size=(512, 512), mode='bilinear') * self.mask
        # tex_cond = self.tex_conv(tex_cond)
        # joint_cond = th.cat([verts_cond, tex_cond], dim=1)
        joint_cond = verts_cond
        x = self.joint_conv_blocks(joint_cond)
        x = x.reshape(B, -1)
        embs_mu = self.mu(x)
        embs_logvar = self.logvar_scale * self.logvar(x)

        # NOTE: the noise is only applied to the input-conditioned values
        if self.training:
            noise = th.randn_like(embs_mu)
            embs = embs_mu + th.exp(embs_logvar) * noise * self.noise_std
        else:
            embs = embs_mu.clone()

        preds.update(
            embs=embs,
            embs_mu=embs_mu,
            embs_logvar=embs_logvar,
        )

        return preds


class ConvDecoder(nn.Module):
    """Multi-region view-independent decoder."""

    def __init__(
        self,
        geo_fn,
        uv_size,
        seam_sampler,
        init_uv_size,
        n_pose_dims,
        n_pose_enc_channels,
        n_embs,
        n_embs_enc_channels,
        n_face_embs,
        n_init_channels,
        n_min_channels,
        assets,
    ):
        super().__init__()

        self.geo_fn = geo_fn

        self.uv_size = uv_size
        self.init_uv_size = init_uv_size
        self.n_pose_dims = n_pose_dims
        self.n_pose_enc_channels = n_pose_enc_channels
        self.n_embs = n_embs
        self.n_embs_enc_channels = n_embs_enc_channels
        self.n_face_embs = n_face_embs

        self.n_blocks = int(np.log2(self.uv_size // init_uv_size))
        self.sizes = [init_uv_size * 2**s for s in range(self.n_blocks + 1)]

        # TODO: just specify a sequence?
        self.n_channels = [
            max(n_init_channels // 2**b, n_min_channels) for b in range(self.n_blocks + 1)
        ]

        logger.info(f"ConvDecoder: n_channels = {self.n_channels}")

        self.local_pose_conv_block = ConvBlock(
            n_pose_dims,
            n_pose_enc_channels,
            init_uv_size,
            kernel_size=1,
            padding=0,
        )

        self.embs_fc = nn.Sequential(
            la.LinearWN(n_embs, 4 * 4 * 128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # TODO: should we switch to the basic version?
        self.embs_conv_block = nn.Sequential(
            UpConvBlockDeep(128, 128, 8),
            UpConvBlockDeep(128, 128, 16),
            UpConvBlockDeep(128, 64, 32),
            UpConvBlockDeep(64, n_embs_enc_channels, 64),
        )

        self.face_embs_fc = nn.Sequential(
            la.LinearWN(n_face_embs, 4 * 4 * 32),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.face_embs_conv_block = nn.Sequential(
            UpConvBlockDeep(32, 64, 8),
            UpConvBlockDeep(64, 64, 16),
            UpConvBlockDeep(64, n_embs_enc_channels, 32),
        )

        n_groups = 2

        self.joint_conv_block = ConvBlock(
            n_pose_enc_channels + n_embs_enc_channels,
            n_init_channels,
            self.init_uv_size,
        )

        self.conv_blocks = nn.ModuleList([])
        for b in range(self.n_blocks):
            self.conv_blocks.append(
                UpConvBlockDeep(
                    self.n_channels[b] * n_groups,
                    self.n_channels[b + 1] * n_groups,
                    self.sizes[b + 1],
                    groups=n_groups,
                ),
            )

        self.verts_conv = la.Conv2dWNUB(
            in_channels=self.n_channels[-1],
            out_channels=3,
            kernel_size=3,
            height=self.uv_size,
            width=self.uv_size,
            padding=1,
        )
        self.tex_conv = la.Conv2dWNUB(
            in_channels=self.n_channels[-1],
            out_channels=3,
            kernel_size=3,
            height=self.uv_size,
            width=self.uv_size,
            padding=1,
        )

        self.apply(weights_initializer(0.2))
        self.verts_conv.apply(weights_initializer(1.0))
        self.tex_conv.apply(weights_initializer(1.0))

        self.seam_sampler = seam_sampler

        # NOTE: removing head region from pose completely
        pose_cond_mask = th.as_tensor(
            assets.pose_cond_mask[np.newaxis] * (1 - assets.head_cond_mask[np.newaxis, np.newaxis]),
            dtype=th.int32,
        )
        self.register_buffer("pose_cond_mask", pose_cond_mask)
        face_cond_mask = th.as_tensor(assets.face_cond_mask, dtype=th.float32)[
            np.newaxis, np.newaxis
        ]
        self.register_buffer("face_cond_mask", face_cond_mask)

        body_cond_mask = th.as_tensor(assets.body_cond_mask, dtype=th.float32)[
            np.newaxis, np.newaxis
        ]
        self.register_buffer("body_cond_mask", body_cond_mask)

    def forward(self, motion, embs, face_embs, embs_conv: Optional[th.Tensor] = None):

        # processing pose
        pose = motion[:, 6:]

        B = pose.shape[0]

        non_head_mask = (self.body_cond_mask * (1.0 - self.face_cond_mask)).clip(0.0, 1.0)

        pose_masked = tile2d(pose, self.init_uv_size) * self.pose_cond_mask
        pose_conv = self.local_pose_conv_block(pose_masked) * non_head_mask

        # TODO: decoding properly?
        if embs_conv is None:
            embs_conv = self.embs_conv_block(self.embs_fc(embs).reshape(B, 128, 4, 4))

        face_conv = self.face_embs_conv_block(self.face_embs_fc(face_embs).reshape(B, 32, 4, 4))
        # merging embeddings with spatial masks
        embs_conv[:, :, 32:, :32] = (
            face_conv * self.face_cond_mask[:, :, 32:, :32]
            + embs_conv[:, :, 32:, :32] * non_head_mask[:, :, 32:, :32]
        )

        joint = th.cat([pose_conv, embs_conv], axis=1)
        joint = self.joint_conv_block(joint)

        x = th.cat([joint, joint], axis=1)
        for b in range(self.n_blocks):
            x = self.conv_blocks[b](x)

        # NOTE: here we do resampling at feature level
        x = self.seam_sampler.impaint(x)
        x = self.seam_sampler.resample(x)
        x = self.seam_sampler.resample(x)

        verts_features, tex_features = th.split(x, self.n_channels[-1], 1)

        verts_uv_delta_rec = self.verts_conv(verts_features)
        # TODO: need to get values
        verts_delta_rec = self.geo_fn.from_uv(verts_uv_delta_rec)
        tex_mean_rec = self.tex_conv(tex_features)

        preds = {
            'geom_delta_rec': verts_delta_rec,
            'geom_uv_delta_rec': verts_uv_delta_rec,
            'tex_mean_rec': tex_mean_rec,
            'embs_conv': embs_conv,
            'pose_conv': pose_conv,
        }

        return preds


class FaceEncoder(nn.Module):
    """A joint encoder for tex and geometry."""

    def __init__(
        self,
        noise_std,
        assets,
        n_embs=256,
        uv_size=512,
        logvar_scale=0.1,
        n_vert_in=7306 * 3,
        prefix="face_",
    ):

        """Fixed-width conv encoder."""
        super().__init__()

        # TODO:
        self.noise_std = noise_std
        self.n_embs = n_embs
        self.logvar_scale = logvar_scale
        self.prefix = prefix
        self.uv_size = uv_size

        assert self.uv_size == 512

        tex_cond_mask = assets.mugsy_face_mask[..., 0]
        tex_cond_mask = th.as_tensor(tex_cond_mask, dtype=th.float32)[np.newaxis, np.newaxis]
        tex_cond_mask = F.interpolate(
            tex_cond_mask, (self.uv_size, self.uv_size), mode="bilinear", align_corners=True
        )
        self.register_buffer("tex_cond_mask", tex_cond_mask)

        self.conv_blocks = nn.Sequential(
            ConvDownBlock(3, 4, 512),
            ConvDownBlock(4, 8, 256),
            ConvDownBlock(8, 16, 128),
            ConvDownBlock(16, 32, 64),
            ConvDownBlock(32, 64, 32),
            ConvDownBlock(64, 128, 16),
            ConvDownBlock(128, 128, 8),
        )
        self.geommod = nn.Sequential(la.LinearWN(n_vert_in, 256), nn.LeakyReLU(0.2, inplace=True))
        self.jointmod = nn.Sequential(
            la.LinearWN(256 + 128 * 4 * 4, 512), nn.LeakyReLU(0.2, inplace=True)
        )
        # TODO: should we put initializer
        self.mu = la.LinearWN(512, self.n_embs)
        self.logvar = la.LinearWN(512, self.n_embs)

        self.apply(weights_initializer(0.2))
        self.mu.apply(weights_initializer(1.0))
        self.logvar.apply(weights_initializer(1.0))

    # TODO: compute_losses()?

    def forward(self, face_geom: th.Tensor, face_tex: th.Tensor, **kwargs):
        B = face_geom.shape[0]

        tex_cond = F.interpolate(
            face_tex, (self.uv_size, self.uv_size), mode="bilinear", align_corners=False
        )
        tex_cond = (tex_cond / 255.0 - 0.5) * self.tex_cond_mask
        x = self.conv_blocks(tex_cond)
        tex_enc = x.reshape(B, 4 * 4 * 128)

        geom_enc = self.geommod(face_geom.reshape(B, -1))

        x = self.jointmod(th.cat([tex_enc, geom_enc], dim=1))
        embs_mu = self.mu(x)
        embs_logvar = self.logvar_scale * self.logvar(x)

        # NOTE: the noise is only applied to the input-conditioned values
        if self.training:
            noise = th.randn_like(embs_mu)
            embs = embs_mu + th.exp(embs_logvar) * noise * self.noise_std
        else:
            embs = embs_mu.clone()

        preds = {"embs": embs, "embs_mu": embs_mu, "embs_logvar": embs_logvar, "tex_cond": tex_cond}
        preds = {f"{self.prefix}{k}": v for k, v in preds.items()}
        return preds


class UNetViewDecoder(nn.Module):
    def __init__(self, geo_fn, net_uv_size, seam_sampler, n_init_ftrs=8):
        super().__init__()
        self.geo_fn = geo_fn
        self.net_uv_size = net_uv_size
        self.unet = UNetWB(4, 3, n_init_ftrs=n_init_ftrs, size=net_uv_size)
        self.register_buffer("faces", self.geo_fn.vi.to(th.int64), persistent=False)

    def forward(self, geom_rec, tex_mean_rec, camera_pos):

        with th.no_grad():
            view_cos = compute_view_cos(geom_rec, self.faces, camera_pos)
            view_cos_uv = self.geo_fn.to_uv(view_cos[..., np.newaxis])
        cond_view = th.cat([view_cos_uv, tex_mean_rec], dim=1)
        tex_view = self.unet(cond_view)
        # TODO: should we try warping here?
        return {"tex_view_rec": tex_view, "cond_view": cond_view}


class UpscaleNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_ftrs, size=1024, upscale_factor=2):
        super().__init__()

        self.conv_block = nn.Sequential(
            la.Conv2dWNUB(in_channels, n_ftrs, size, size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.out_block = la.Conv2dWNUB(
            n_ftrs,
            out_channels * upscale_factor**2,
            size,
            size,
            kernel_size=1,
            padding=0,
        )

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=upscale_factor)
        self.apply(weights_initializer(0.2))
        self.out_block.apply(weights_initializer(1.0))

    def forward(self, x):
        x = self.conv_block(x)
        x = self.out_block(x)
        return self.pixel_shuffle(x)