"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import json
from typing import Callable, Optional

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from model.guide import GuideTransformer
from model.modules.audio_encoder import Wav2VecEncoder
from model.modules.rotary_embedding_torch import RotaryEmbedding
from model.modules.transformer_modules import (
    DecoderLayerStack,
    FiLMTransformerDecoderLayer,
    RegressionTransformer,
    TransformerEncoderLayerRotary,
)
from model.utils import (
    init_weight,
    PositionalEncoding,
    prob_mask_like,
    setup_lip_regressor,
    SinusoidalPosEmb,
)
from model.vqvae import setup_tokenizer
from torch.nn import functional as F
from utils.misc import prGreen, prRed


class Audio2LipRegressionTransformer(torch.nn.Module):
    def __init__(
        self,
        n_vertices: int = 338,
        causal: bool = False,
        train_wav2vec: bool = False,
        transformer_encoder_layers: int = 2,
        transformer_decoder_layers: int = 4,
    ):
        super().__init__()
        self.n_vertices = n_vertices

        self.audio_encoder = Wav2VecEncoder()
        if not train_wav2vec:
            self.audio_encoder.eval()
            for param in self.audio_encoder.parameters():
                param.requires_grad = False

        self.regression_model = RegressionTransformer(
            transformer_encoder_layers=transformer_encoder_layers,
            transformer_decoder_layers=transformer_decoder_layers,
            d_model=512,
            d_cond=512,
            num_heads=4,
            causal=causal,
        )
        self.project_output = torch.nn.Linear(512, self.n_vertices * 3)

    def forward(self, audio):
        """
        :param audio: tensor of shape B x T x 1600
        :return: tensor of shape B x T x n_vertices x 3 containing reconstructed lip geometry
        """
        B, T = audio.shape[0], audio.shape[1]

        cond = self.audio_encoder(audio)

        x = torch.zeros(B, T, 512, device=audio.device)
        x = self.regression_model(x, cond)
        x = self.project_output(x)

        verts = x.view(B, T, self.n_vertices, 3)
        return verts


class FiLMTransformer(nn.Module):
    def __init__(
        self,
        args,
        nfeats: int,
        latent_dim: int = 512,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        cond_feature_dim: int = 4800,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
        use_rotary: bool = True,
        cond_mode: str = "audio",
        split_type: str = "train",
        device: str = "cuda",
        **kwargs,
    ) -> None:
        super().__init__()
        self.nfeats = nfeats
        self.cond_mode = cond_mode
        self.cond_feature_dim = cond_feature_dim
        self.add_frame_cond = args.add_frame_cond
        self.data_format = args.data_format
        self.split_type = split_type
        self.device = device

        # positional embeddings
        self.rotary = None
        self.abs_pos_encoding = nn.Identity()
        # if rotary, replace absolute embedding with a rotary embedding instance (absolute becomes an identity)
        if use_rotary:
            self.rotary = RotaryEmbedding(dim=latent_dim)
        else:
            self.abs_pos_encoding = PositionalEncoding(
                latent_dim, dropout, batch_first=True
            )

        # time embedding processing
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(latent_dim),
            nn.Linear(latent_dim, latent_dim * 4),
            nn.Mish(),
        )
        self.to_time_cond = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim),
        )
        self.to_time_tokens = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim * 2),
            Rearrange("b (r d) -> b r d", r=2),
        )

        # null embeddings for guidance dropout
        self.seq_len = args.max_seq_length
        emb_len = 1998  # hardcoded for now
        self.null_cond_embed = nn.Parameter(torch.randn(1, emb_len, latent_dim))
        self.null_cond_hidden = nn.Parameter(torch.randn(1, latent_dim))
        self.norm_cond = nn.LayerNorm(latent_dim)
        self.setup_audio_models()

        # set up pose/face specific parts of the model
        self.input_projection = nn.Linear(self.nfeats, latent_dim)
        if self.data_format == "pose":
            cond_feature_dim = 1024
            key_feature_dim = 104
            self.step = 30
            self.use_cm = True
            self.setup_guide_models(args, latent_dim, key_feature_dim)
            self.post_pose_layers = self._build_single_pose_conv(self.nfeats)
            self.post_pose_layers.apply(init_weight)
            self.final_conv = torch.nn.Conv1d(self.nfeats, self.nfeats, kernel_size=1)
            self.receptive_field = 25
        elif self.data_format == "face":
            self.use_cm = False
            cond_feature_dim = 1024 + 1014
            self.setup_lip_models()
            self.cond_encoder = nn.Sequential()
            for _ in range(2):
                self.cond_encoder.append(
                    TransformerEncoderLayerRotary(
                        d_model=latent_dim,
                        nhead=num_heads,
                        dim_feedforward=ff_size,
                        dropout=dropout,
                        activation=activation,
                        batch_first=True,
                        rotary=self.rotary,
                    )
                )
            self.cond_encoder.apply(init_weight)

        self.cond_projection = nn.Linear(cond_feature_dim, latent_dim)
        self.non_attn_cond_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # decoder
        decoderstack = nn.ModuleList([])
        for _ in range(num_layers):
            decoderstack.append(
                FiLMTransformerDecoderLayer(
                    latent_dim,
                    num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                    use_cm=self.use_cm,
                )
            )
        self.seqTransDecoder = DecoderLayerStack(decoderstack)
        self.seqTransDecoder.apply(init_weight)
        self.final_layer = nn.Linear(latent_dim, self.nfeats)
        self.final_layer.apply(init_weight)

    def _build_single_pose_conv(self, nfeats: int) -> nn.ModuleList:
        post_pose_layers = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(nfeats, max(256, nfeats), kernel_size=3, dilation=1),
                torch.nn.Conv1d(max(256, nfeats), nfeats, kernel_size=3, dilation=2),
                torch.nn.Conv1d(nfeats, nfeats, kernel_size=3, dilation=3),
                torch.nn.Conv1d(nfeats, nfeats, kernel_size=3, dilation=1),
                torch.nn.Conv1d(nfeats, nfeats, kernel_size=3, dilation=2),
                torch.nn.Conv1d(nfeats, nfeats, kernel_size=3, dilation=3),
            ]
        )
        return post_pose_layers

    def _run_single_pose_conv(self, output: torch.Tensor) -> torch.Tensor:
        output = torch.nn.functional.pad(output, pad=[self.receptive_field - 1, 0])
        for _, layer in enumerate(self.post_pose_layers):
            y = torch.nn.functional.leaky_relu(layer(output), negative_slope=0.2)
            if self.split_type == "train":
                y = torch.nn.functional.dropout(y, 0.2)
            if output.shape[1] == y.shape[1]:
                output = (output[:, :, -y.shape[-1] :] + y) / 2.0  # skip connection
            else:
                output = y
        return output

    def setup_guide_models(self, args, latent_dim: int, key_feature_dim: int) -> None:
        # set up conditioning info
        max_keyframe_len = len(list(range(self.seq_len))[:: self.step])
        self.null_pose_embed = nn.Parameter(
            torch.randn(1, max_keyframe_len, latent_dim)
        )
        prGreen(f"using keyframes: {self.null_pose_embed.shape}")
        self.frame_cond_projection = nn.Linear(key_feature_dim, latent_dim)
        self.frame_norm_cond = nn.LayerNorm(latent_dim)
        # for test time set up keyframe transformer
        self.resume_trans = None
        if self.split_type == "test":
            if hasattr(args, "resume_trans") and args.resume_trans is not None:
                self.resume_trans = args.resume_trans
                self.setup_guide_predictor(args.resume_trans)
            else:
                prRed("not using transformer, just using ground truth")

    def setup_guide_predictor(self, cp_path: str) -> None:
        cp_dir = cp_path.split("checkpoints/iter-")[0]
        with open(f"{cp_dir}/args.json") as f:
            trans_args = json.load(f)

        # set up tokenizer based on trans_arg load point
        self.tokenizer = setup_tokenizer(trans_args["resume_pth"])

        # set up transformer
        self.transformer = GuideTransformer(
            tokens=self.tokenizer.n_clusters,
            num_layers=trans_args["layers"],
            dim=trans_args["dim"],
            emb_len=1998,
            num_audio_layers=trans_args["num_audio_layers"],
        )
        for param in self.transformer.parameters():
            param.requires_grad = False
        prGreen("loading TRANSFORMER checkpoint from {}".format(cp_path))
        cp = torch.load(cp_path)
        missing_keys, unexpected_keys = self.transformer.load_state_dict(
            cp["model_state_dict"], strict=False
        )
        assert len(missing_keys) == 0, missing_keys
        assert len(unexpected_keys) == 0, unexpected_keys

    def setup_audio_models(self) -> None:
        self.audio_model, self.audio_resampler = setup_lip_regressor()

    def setup_lip_models(self) -> None:
        self.lip_model = Audio2LipRegressionTransformer()
        cp_path = "./assets/iter-0200000.pt"
        cp = torch.load(cp_path, map_location=torch.device(self.device))
        self.lip_model.load_state_dict(cp["model_state_dict"])
        for param in self.lip_model.parameters():
            param.requires_grad = False
        prGreen(f"adding lip conditioning {cp_path}")

    def guided_forward(
        self,
        x: torch.Tensor,
        cond_embed: torch.Tensor,
        times: torch.Tensor,
        guidance_weight: float,
    ) -> torch.Tensor:
        unc = self.forward(x, cond_embed, times, cond_drop_prob=1)
        conditioned = self.forward(x, cond_embed, times, cond_drop_prob=0)
        return unc + (conditioned - unc) * guidance_weight

    def parameters_w_grad(self):
        return [p for p in self.parameters() if p.requires_grad]

    def encode_audio(self, raw_audio: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        a0 = self.audio_resampler(raw_audio[:, :, 0].to(device))
        a1 = self.audio_resampler(raw_audio[:, :, 1].to(device))
        with torch.no_grad():
            z0 = self.audio_model.feature_extractor(a0)
            z1 = self.audio_model.feature_extractor(a1)
            emb = torch.cat((z0, z1), axis=1).permute(0, 2, 1)
        return emb

    def encode_lip(self, audio: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        reshaped_audio = audio.reshape((audio.shape[0], -1, 1600, 2))[..., 0]
        # processes 4 seconds at a time
        B, T, _ = reshaped_audio.shape
        lip_cond = torch.zeros(
            (audio.shape[0], T, 338, 3),
            device=audio.device,
            dtype=audio.dtype,
        )
        for i in range(0, T, 120):
            lip_cond[:, i : i + 120, ...] = self.lip_model(
                reshaped_audio[:, i : i + 120, ...]
            )
        lip_cond = lip_cond.permute(0, 2, 3, 1).reshape((B, 338 * 3, -1))
        lip_cond = torch.nn.functional.interpolate(
            lip_cond, size=cond_embed.shape[1], mode="nearest-exact"
        ).permute(0, 2, 1)
        cond_embed = torch.cat((cond_embed, lip_cond), dim=-1)
        return cond_embed

    def encode_keyframes(
        self, y: torch.Tensor, cond_drop_prob: float, batch_size: int
    ) -> torch.Tensor:
        pred = y["keyframes"]
        new_mask = y["mask"][..., :: self.step].squeeze((1, 2))
        pred[~new_mask] = 0.0  # pad the unknown
        pose_hidden = self.frame_cond_projection(pred.detach().clone().cuda())
        pose_embed = self.abs_pos_encoding(pose_hidden)
        pose_tokens = self.frame_norm_cond(pose_embed)
        # do conditional dropout for guide poses
        key_cond_drop_prob = cond_drop_prob
        keep_mask_pose = prob_mask_like(
            (batch_size,), 1 - key_cond_drop_prob, device=pose_tokens.device
        )
        keep_mask_pose_embed = rearrange(keep_mask_pose, "b -> b 1 1")
        null_pose_embed = self.null_pose_embed.to(pose_tokens.dtype)
        pose_tokens = torch.where(
            keep_mask_pose_embed,
            pose_tokens,
            null_pose_embed[:, : pose_tokens.shape[1], :],
        )
        return pose_tokens

    def forward(
        self,
        x: torch.Tensor,
        times: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        cond_drop_prob: float = 0.0,
    ) -> torch.Tensor:
        if x.dim() == 4:
            x = x.permute(0, 3, 1, 2).squeeze(-1)
        batch_size, device = x.shape[0], x.device
        if self.cond_mode == "uncond":
            cond_embed = torch.zeros(
                (x.shape[0], x.shape[1], self.cond_feature_dim),
                dtype=x.dtype,
                device=x.device,
            )
        else:
            cond_embed = y["audio"]
            cond_embed = self.encode_audio(cond_embed)
            if self.data_format == "face":
                cond_embed = self.encode_lip(y["audio"], cond_embed)
                pose_tokens = None
            if self.data_format == "pose":
                pose_tokens = self.encode_keyframes(y, cond_drop_prob, batch_size)
        assert cond_embed is not None, "cond emb should not be none"
        # process conditioning information
        x = self.input_projection(x)
        x = self.abs_pos_encoding(x)
        audio_cond_drop_prob = cond_drop_prob
        keep_mask = prob_mask_like(
            (batch_size,), 1 - audio_cond_drop_prob, device=device
        )
        keep_mask_embed = rearrange(keep_mask, "b -> b 1 1")
        keep_mask_hidden = rearrange(keep_mask, "b -> b 1")
        cond_tokens = self.cond_projection(cond_embed)
        cond_tokens = self.abs_pos_encoding(cond_tokens)
        if self.data_format == "face":
            cond_tokens = self.cond_encoder(cond_tokens)
        null_cond_embed = self.null_cond_embed.to(cond_tokens.dtype)
        cond_tokens = torch.where(
            keep_mask_embed, cond_tokens, null_cond_embed[:, : cond_tokens.shape[1], :]
        )
        mean_pooled_cond_tokens = cond_tokens.mean(dim=-2)
        cond_hidden = self.non_attn_cond_projection(mean_pooled_cond_tokens)

        # create t conditioning
        t_hidden = self.time_mlp(times)
        t = self.to_time_cond(t_hidden)
        t_tokens = self.to_time_tokens(t_hidden)
        null_cond_hidden = self.null_cond_hidden.to(t.dtype)
        cond_hidden = torch.where(keep_mask_hidden, cond_hidden, null_cond_hidden)
        t += cond_hidden

        # cross-attention conditioning
        c = torch.cat((cond_tokens, t_tokens), dim=-2)
        cond_tokens = self.norm_cond(c)

        # Pass through the transformer decoder
        output = self.seqTransDecoder(x, cond_tokens, t, memory2=pose_tokens)
        output = self.final_layer(output)
        if self.data_format == "pose":
            output = output.permute(0, 2, 1)
            output = self._run_single_pose_conv(output)
            output = self.final_conv(output)
            output = output.permute(0, 2, 1)
        return output
