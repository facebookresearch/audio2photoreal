"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Callable, List

import torch
import torch as th
import torch.nn as nn
from einops import rearrange
from model.modules.rotary_embedding_torch import RotaryEmbedding

from model.modules.transformer_modules import (
    DecoderLayerStack,
    FiLMTransformerDecoderLayer,
    PositionalEncoding,
)
from model.utils import prob_mask_like, setup_lip_regressor
from torch.distributions import Categorical
from torch.nn import functional as F


class GuideTransformer(nn.Module):
    def __init__(
        self,
        tokens: int,
        num_heads: int = 4,
        num_layers: int = 4,
        dim: int = 512,
        ff_size: int = 1024,
        dropout: float = 0.1,
        activation: Callable = F.gelu,
        use_rotary: bool = True,
        cond_feature_dim: int = 1024,
        emb_len: int = 798,
        num_audio_layers: int = 2,
    ):
        super().__init__()
        self.tokens = tokens
        self.token_embedding = th.nn.Embedding(
            num_embeddings=tokens + 1,  # account for sequence start and end tokens
            embedding_dim=dim,
        )
        self.abs_pos_encoding = nn.Identity()
        # if rotary, replace absolute embedding with a rotary embedding instance (absolute becomes an identity)
        if use_rotary:
            self.rotary = RotaryEmbedding(dim=dim)
        else:
            self.abs_pos_encoding = PositionalEncoding(dim, dropout, batch_first=True)
        self.setup_audio_models(cond_feature_dim, num_audio_layers)

        self.null_cond_embed = nn.Parameter(torch.randn(1, emb_len, dim))
        self.null_cond_hidden = nn.Parameter(torch.randn(1, dim))
        self.norm_cond = nn.LayerNorm(dim)

        self.cond_projection = nn.Linear(cond_feature_dim, dim)
        self.non_attn_cond_projection = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        # decoder
        decoderstack = nn.ModuleList([])
        for _ in range(num_layers):
            decoderstack.append(
                FiLMTransformerDecoderLayer(
                    dim,
                    num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )
        self.seqTransDecoder = DecoderLayerStack(decoderstack)
        self.final_layer = nn.Linear(dim, tokens)

    def _build_single_audio_conv(self, c: int) -> List[nn.Module]:
        return [
            torch.nn.Conv1d(c, max(256, c), kernel_size=3, dilation=1),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Dropout(0.2),
            #
            torch.nn.Conv1d(max(256, c), max(256, c), kernel_size=3, dilation=2),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Dropout(0.2),
            #
            torch.nn.Conv1d(max(128, c), max(128, c), kernel_size=3, dilation=3),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Dropout(0.2),
            #
            torch.nn.Conv1d(max(128, c), c, kernel_size=3, dilation=1),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Dropout(0.2),
            #
            torch.nn.Conv1d(c, c, kernel_size=3, dilation=2),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Dropout(0.2),
            #
            torch.nn.Conv1d(c, c, kernel_size=3, dilation=3),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Dropout(0.2),
        ]

    def setup_audio_models(self, cond_feature_dim: int, num_audio_layers: int) -> None:
        pre_layers = []
        for _ in range(num_audio_layers):
            pre_layers += self._build_single_audio_conv(cond_feature_dim)
        pre_layers += [
            torch.nn.Conv1d(cond_feature_dim, cond_feature_dim, kernel_size=1)
        ]
        pre_layers = torch.nn.ModuleList(pre_layers)
        self.pre_audio = nn.Sequential(*pre_layers)
        self.audio_model, self.audio_resampler = setup_lip_regressor()

    def encode_audio(self, raw_audio: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        a0 = self.audio_resampler(raw_audio[:, :, 0].to(device))  # B x T
        a1 = self.audio_resampler(raw_audio[:, :, 1].to(device))  # B x T
        with torch.no_grad():
            z0 = self.audio_model.feature_extractor(a0)
            z1 = self.audio_model.feature_extractor(a1)
            emb = torch.cat((z0, z1), axis=1).permute(0, 2, 1)
        return emb

    def get_tgt_mask(self, size: int, device: str) -> torch.tensor:
        mask = torch.tril(
            torch.ones((size, size), device=device) == 1
        )  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float("-inf"))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0
        return mask

    def forward(
        self, tokens: th.Tensor, condition: th.Tensor, cond_drop_prob: float = 0.0
    ) -> torch.Tensor:
        batch_size, device = tokens.shape[0], tokens.device

        x = self.token_embedding(tokens)
        x = self.abs_pos_encoding(x)
        tgt_mask = self.get_tgt_mask(x.shape[1], x.device)

        cond_embed = self.encode_audio(condition)
        keep_mask = prob_mask_like((batch_size,), 1 - cond_drop_prob, device=device)
        keep_mask_embed = rearrange(keep_mask, "b -> b 1 1")
        keep_mask_hidden = rearrange(keep_mask, "b -> b 1")
        cond_tokens = self.pre_audio(cond_embed.permute(0, 2, 1)).permute(0, 2, 1)
        #
        cond_tokens = self.cond_projection(cond_tokens)
        cond_tokens = self.abs_pos_encoding(cond_tokens)

        null_cond_embed = self.null_cond_embed.to(cond_tokens.dtype)
        cond_tokens = torch.where(
            keep_mask_embed, cond_tokens, null_cond_embed[:, : cond_tokens.shape[1], :]
        )
        mean_pooled_cond_tokens = cond_tokens.mean(dim=-2)
        cond_hidden = self.non_attn_cond_projection(mean_pooled_cond_tokens)

        # FiLM conditioning
        null_cond_hidden = self.null_cond_hidden.to(cond_tokens.dtype)
        cond_hidden = torch.where(keep_mask_hidden, cond_hidden, null_cond_hidden)
        cond_tokens = self.norm_cond(cond_tokens)

        output = self.seqTransDecoder(x, cond_tokens, cond_hidden, tgt_mask=tgt_mask)
        output = self.final_layer(output)
        return output

    def generate(
        self,
        condition: th.Tensor,
        sequence_length: int,
        layers: int,
        n_sequences: int = 1,
        max_key_len: int = 8,
        max_seq_len: int = 240,
        top_p: float = 0.94,
    ) -> torch.Tensor:
        """
        :param sequence_length: number of tokens to generate in autoregressive fashion
        :param n_sequences: number of sequences to generate simultaneously
        :param temperature: temerature of the softmax for sampling from the output logits
        :return n_sequences x sequence_length LongTensor containing generated tokens
        """
        assert max_key_len == int(max_seq_len / 30), "currently only running for 1fps"
        max_key_len *= layers
        with th.no_grad():
            input_tokens = (
                th.zeros(n_sequences, 1, dtype=th.int64).to(condition.device)
                + self.tokens
            )
            for _ in range(sequence_length * layers):
                curr_input_tokens = input_tokens
                curr_condition = condition
                logits = self.forward(curr_input_tokens, curr_condition)
                logits = logits[:, -1, :]  # only most recent time step is relevant
                one_hot = th.nn.functional.softmax(logits, dim=-1)
                sorted_probs, indices = torch.sort(one_hot, dim=-1, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                nucleus = cumulative_probs < top_p
                nucleus = torch.cat(
                    [
                        nucleus.new_ones(nucleus.shape[:-1] + (1,)),
                        nucleus[..., :-1],
                    ],
                    dim=-1,
                )
                sorted_probs[~nucleus] = 0
                sorted_probs /= sorted_probs.sum(-1, keepdim=True)
                dist = Categorical(sorted_probs)
                idx = dist.sample()
                tokens = indices.gather(-1, idx.unsqueeze(-1))
                input_tokens = th.cat([input_tokens, tokens], dim=-1)

            # return generated tokens except for sequence start token
            tokens = input_tokens[:, 1:].contiguous()
            return tokens
