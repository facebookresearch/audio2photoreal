"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import math
from typing import Any, Callable, List, Optional, Union

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F


def generate_causal_mask(source_length, target_length, device="cpu"):
    if source_length == target_length:
        mask = (
            torch.triu(torch.ones(target_length, source_length, device=device)) == 1
        ).transpose(0, 1)
    else:
        mask = torch.zeros(target_length, source_length, device=device)
        idx = torch.linspace(0, source_length, target_length + 1)[1:].round().long()
        for i in range(target_length):
            mask[i, 0 : idx[i]] = 1

    return (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )


class TransformerEncoderLayerRotary(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = True,
        rotary=None,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

        self.rotary = rotary
        self.use_rotary = rotary is not None

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class DenseFiLM(nn.Module):
    """Feature-wise linear modulation (FiLM) generator."""

    def __init__(self, embed_channels):
        super().__init__()
        self.embed_channels = embed_channels
        self.block = nn.Sequential(
            nn.Mish(), nn.Linear(embed_channels, embed_channels * 2)
        )

    def forward(self, position):
        pos_encoding = self.block(position)
        pos_encoding = rearrange(pos_encoding, "b c -> b 1 c")
        scale_shift = pos_encoding.chunk(2, dim=-1)
        return scale_shift


def featurewise_affine(x, scale_shift):
    scale, shift = scale_shift
    return (scale + 1) * x + shift


class FiLMTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        batch_first=False,
        norm_first=True,
        rotary=None,
        use_cm=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation

        self.film1 = DenseFiLM(d_model)
        self.film2 = DenseFiLM(d_model)
        self.film3 = DenseFiLM(d_model)

        if use_cm:
            self.multihead_attn2 = nn.MultiheadAttention(  # 2
                d_model, nhead, dropout=dropout, batch_first=batch_first
            )
            self.norm2a = nn.LayerNorm(d_model, eps=layer_norm_eps)  # 2
            self.dropout2a = nn.Dropout(dropout)  # 2
            self.film2a = DenseFiLM(d_model)  # 2

        self.rotary = rotary
        self.use_rotary = rotary is not None

    # x, cond, t
    def forward(
        self,
        tgt,
        memory,
        t,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        memory2=None,
    ):
        x = tgt
        if self.norm_first:
            # self-attention -> film -> residual
            x_1 = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + featurewise_affine(x_1, self.film1(t))
            # cross-attention -> film -> residual
            x_2 = self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                self.multihead_attn,
                self.dropout2,
            )
            x = x + featurewise_affine(x_2, self.film2(t))
            if memory2 is not None:
                # cross-attention x2 -> film -> residual
                x_2a = self._mha_block(
                    self.norm2a(x),
                    memory2,
                    memory_mask,
                    memory_key_padding_mask,
                    self.multihead_attn2,
                    self.dropout2a,
                )
                x = x + featurewise_affine(x_2a, self.film2a(t))
            # feedforward -> film -> residual
            x_3 = self._ff_block(self.norm3(x))
            x = x + featurewise_affine(x_3, self.film3(t))
        else:
            x = self.norm1(
                x
                + featurewise_affine(
                    self._sa_block(x, tgt_mask, tgt_key_padding_mask), self.film1(t)
                )
            )
            x = self.norm2(
                x
                + featurewise_affine(
                    self._mha_block(x, memory, memory_mask, memory_key_padding_mask),
                    self.film2(t),
                )
            )
            x = self.norm3(x + featurewise_affine(self._ff_block(x), self.film3(t)))
        return x

    # self-attention block
    # qkv
    def _sa_block(self, x, attn_mask, key_padding_mask):
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # multihead attention block
    # qkv
    def _mha_block(self, x, mem, attn_mask, key_padding_mask, mha, dropout):
        q = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        k = self.rotary.rotate_queries_or_keys(mem) if self.use_rotary else mem
        x = mha(
            q,
            k,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return dropout(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class DecoderLayerStack(nn.Module):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack

    def forward(self, x, cond, t, tgt_mask=None, memory2=None):
        for layer in self.stack:
            x = layer(x, cond, t, tgt_mask=tgt_mask, memory2=memory2)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor):
        """
        :param x: B x T x d_model tensor
        :return: B x T x d_model tensor
        """
        x = x + self.pe[None, : x.shape[1], :]
        x = self.dropout(x)
        return x


class TimestepEncoding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()

        # Fourier embedding
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        self.register_buffer("emb", emb)

        # encoding
        self.encoding = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.Mish(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )

    def forward(self, t: torch.Tensor):
        """
        :param t: B-dimensional tensor containing timesteps in range [0, 1]
        :return: B x embedding_dim tensor containing timestep encodings
        """
        x = t[:, None] * self.emb[None, :]
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        x = self.encoding(x)
        return x


class FiLM(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.film = nn.Sequential(nn.Mish(), nn.Linear(dim, dim * 2))

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        :param x: ... x dim tensor
        :param cond: ... x dim tensor
        :return: ... x dim tensor as scale(cond) * x + bias(cond)
        """
        cond = self.film(cond)
        scale, bias = torch.chunk(cond, chunks=2, dim=-1)
        x = (scale + 1) * x + bias
        return x


class FeedforwardBlock(nn.Module):
    def __init__(self, d_model: int, d_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_feedforward),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_feedforward, d_model),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor):
        """
        :param x: ... x d_model tensor
        :return: ... x d_model tensor
        """
        return self.ff(x)


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
    ):
        """
        :param x: B x T x d_model input tensor
        :param attn_mask: B * num_heads x L x S mask with L=target sequence length, S=source sequence length
                          for a float mask: values will be added to attention weight
                          for a binary mask: True indicates that the element is not allowed to attend
        :param key_padding_mask: B x S mask
                          for a float mask: values will be added directly to the corresponding key values
                          for a binary mask: True indicates that the corresponding key value will be ignored
        :return: B x T x d_model output tensor
        """
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = self.dropout(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, d_cond: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=d_cond,
            vdim=d_cond,
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
    ):
        """
        :param x: B x T_target x d_model input tensor
        :param cond: B x T_cond x d_cond condition tensor
        :param attn_mask: B * num_heads x L x S mask with L=target sequence length, S=source sequence length
                          for a float mask: values will be added to attention weight
                          for a binary mask: True indicates that the element is not allowed to attend
        :param key_padding_mask: B x S mask
                          for a float mask: values will be added directly to the corresponding key values
                          for a binary mask: True indicates that the corresponding key value will be ignored
        :return: B x T x d_model output tensor
        """
        x = self.cross_attn(
            x,
            cond,
            cond,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = SelfAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.feedforward = FeedforwardBlock(d_model, d_feedforward, dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
    ):
        x = x + self.self_attn(self.norm1(x), mask, key_padding_mask)
        x = x + self.feedforward(self.norm2(x))
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_cond: int,
        num_heads: int,
        d_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = SelfAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = CrossAttention(d_model, d_cond, num_heads, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.feedforward = FeedforwardBlock(d_model, d_feedforward, dropout)

    def forward(
        self,
        x: torch.Tensor,
        cross_cond: torch.Tensor,
        target_mask: torch.Tensor = None,
        target_key_padding_mask: torch.Tensor = None,
        cross_cond_mask: torch.Tensor = None,
        cross_cond_key_padding_mask: torch.Tensor = None,
    ):
        """
        :param x: B x T x d_model tensor
        :param cross_cond: B x T x d_cond tensor containing the conditioning input to cross attention layers
        :return: B x T x d_model tensor
        """
        x = x + self.self_attn(self.norm1(x), target_mask, target_key_padding_mask)
        x = x + self.cross_attn(
            self.norm2(x), cross_cond, cross_cond_mask, cross_cond_key_padding_mask
        )
        x = x + self.feedforward(self.norm3(x))
        return x


class FilmTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_cond: int,
        num_heads: int,
        d_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = SelfAttention(d_model, num_heads, dropout)
        self.film1 = FiLM(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = CrossAttention(d_model, d_cond, num_heads, dropout)
        self.film2 = FiLM(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.feedforward = FeedforwardBlock(d_model, d_feedforward, dropout)
        self.film3 = FiLM(d_model)

    def forward(
        self,
        x: torch.Tensor,
        cross_cond: torch.Tensor,
        film_cond: torch.Tensor,
        target_mask: torch.Tensor = None,
        target_key_padding_mask: torch.Tensor = None,
        cross_cond_mask: torch.Tensor = None,
        cross_cond_key_padding_mask: torch.Tensor = None,
    ):
        """
        :param x: B x T x d_model tensor
        :param cross_cond: B x T x d_cond tensor containing the conditioning input to cross attention layers
        :param film_cond: B x [1 or T] x film_cond tensor containing the conditioning input to FiLM layers
        :return: B x T x d_model tensor
        """
        x1 = self.self_attn(self.norm1(x), target_mask, target_key_padding_mask)
        x = x + self.film1(x1, film_cond)
        x2 = self.cross_attn(
            self.norm2(x), cross_cond, cross_cond_mask, cross_cond_key_padding_mask
        )
        x = x + self.film2(x2, film_cond)
        x3 = self.feedforward(self.norm3(x))
        x = x + self.film3(x3, film_cond)
        return x


class RegressionTransformer(nn.Module):
    def __init__(
        self,
        transformer_encoder_layers: int = 2,
        transformer_decoder_layers: int = 4,
        d_model: int = 512,
        d_cond: int = 512,
        num_heads: int = 4,
        d_feedforward: int = 1024,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        self.causal = causal

        self.cond_positional_encoding = PositionalEncoding(d_cond, dropout)
        self.target_positional_encoding = PositionalEncoding(d_model, dropout)

        self.transformer_encoder = nn.ModuleList(
            [
                TransformerEncoderLayer(d_cond, num_heads, d_feedforward, dropout)
                for _ in range(transformer_encoder_layers)
            ]
        )

        self.transformer_decoder = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model, d_cond, num_heads, d_feedforward, dropout
                )
                for _ in range(transformer_decoder_layers)
            ]
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        :param x: B x T x d_model input tensor
        :param cond: B x T x d_cond conditional tensor
        :return: B x T x d_model output tensor
        """
        x = self.target_positional_encoding(x)
        cond = self.cond_positional_encoding(cond)

        if self.causal:
            encoder_mask = generate_causal_mask(
                cond.shape[1], cond.shape[1], device=cond.device
            )
            decoder_self_attn_mask = generate_causal_mask(
                x.shape[1], x.shape[1], device=x.device
            )
            decoder_cross_attn_mask = generate_causal_mask(
                cond.shape[1], x.shape[1], device=x.device
            )
        else:
            encoder_mask = None
            decoder_self_attn_mask = None
            decoder_cross_attn_mask = None

        for encoder_layer in self.transformer_encoder:
            cond = encoder_layer(cond, mask=encoder_mask)
        for decoder_layer in self.transformer_decoder:
            x = decoder_layer(
                x,
                cond,
                target_mask=decoder_self_attn_mask,
                cross_cond_mask=decoder_cross_attn_mask,
            )
        return x


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        transformer_encoder_layers: int = 2,
        transformer_decoder_layers: int = 4,
        d_model: int = 512,
        d_cond: int = 512,
        num_heads: int = 4,
        d_feedforward: int = 1024,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        self.causal = causal

        self.timestep_encoder = TimestepEncoding(d_model)
        self.cond_positional_encoding = PositionalEncoding(d_cond, dropout)
        self.target_positional_encoding = PositionalEncoding(d_model, dropout)

        self.transformer_encoder = nn.ModuleList(
            [
                TransformerEncoderLayer(d_cond, num_heads, d_feedforward, dropout)
                for _ in range(transformer_encoder_layers)
            ]
        )

        self.transformer_decoder = nn.ModuleList(
            [
                FilmTransformerDecoderLayer(
                    d_model, d_cond, num_heads, d_feedforward, dropout
                )
                for _ in range(transformer_decoder_layers)
            ]
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor):
        """
        :param x: B x T x d_model input tensor
        :param cond: B x T x d_cond conditional tensor
        :param t: B-dimensional tensor containing diffusion timesteps in range [0, 1]
        :return: B x T x d_model output tensor
        """
        t = self.timestep_encoder(t).unsqueeze(1)  # B x 1 x d_model
        x = self.target_positional_encoding(x)
        cond = self.cond_positional_encoding(cond)

        if self.causal:
            encoder_mask = generate_causal_mask(
                cond.shape[1], cond.shape[1], device=cond.device
            )
            decoder_self_attn_mask = generate_causal_mask(
                x.shape[1], x.shape[1], device=x.device
            )
            decoder_cross_attn_mask = generate_causal_mask(
                cond.shape[1], x.shape[1], device=x.device
            )
        else:
            encoder_mask = None
            decoder_self_attn_mask = None
            decoder_cross_attn_mask = None

        for encoder_layer in self.transformer_encoder:
            cond = encoder_layer(cond, mask=encoder_mask)
        for decoder_layer in self.transformer_decoder:
            x = decoder_layer(
                x,
                cond,
                t,
                target_mask=decoder_self_attn_mask,
                cross_cond_mask=decoder_cross_attn_mask,
            )

        return x
