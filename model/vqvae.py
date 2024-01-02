"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from utils.misc import broadcast_tensors


def setup_tokenizer(resume_pth: str) -> "TemporalVertexCodec":
    args_path = os.path.dirname(resume_pth)
    with open(os.path.join(args_path, "args.json")) as f:
        trans_args = json.load(f)
    tokenizer = TemporalVertexCodec(
        n_vertices=trans_args["nb_joints"],
        latent_dim=trans_args["output_emb_width"],
        categories=trans_args["code_dim"],
        residual_depth=trans_args["depth"],
    )
    print("loading checkpoint from {}".format(resume_pth))
    ckpt = torch.load(resume_pth, map_location="cpu")
    tokenizer.load_state_dict(ckpt["net"], strict=True)
    for p in tokenizer.parameters():
        p.requires_grad = False
    tokenizer.cuda()
    return tokenizer


def default(val, d):
    return val if val is not None else d


def ema_inplace(moving_avg, new, decay: float):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories: int, epsilon: float = 1e-5):
    return (x + epsilon) / (x.sum() + n_categories * epsilon)


def uniform_init(*shape: int):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def sum_flat(tensor):
    """
    Take the sum over all non-batch dimensions.
    """
    return tensor.sum(dim=list(range(1, len(tensor.shape))))


def sample_vectors(samples, num: int):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def kmeans(samples, num_clusters: int, num_iters: int = 10):
    dim, dtype = samples.shape[-1], samples.dtype

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        diffs = rearrange(samples, "n d -> n () d") - rearrange(means, "c d -> () c d")
        dists = -(diffs**2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


class EuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance.
    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
        kmeans_init (bool): Whether to use k-means to initialize the codebooks.
            If set to true, run the k-means algorithm on the first training batch and use
            the learned centroids as initialization.
        kmeans_iters (int): Number of iterations used for k-means algorithm at initialization.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: int = False,
        kmeans_iters: int = 10,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.decay = decay
        init_fn = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size

        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code

        self.register_buffer("inited", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.inited:
            return

        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.copy_(torch.Tensor([True]))
        # Make sure all buffers across workers are in sync after initialization
        broadcast_tensors(self.buffers())

    def replace_(self, samples, mask):
        modified_codebook = torch.where(
            mask[..., None], sample_vectors(samples, self.codebook_size), self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self.replace_(batch_samples, mask=expired_codes)
        broadcast_tensors(self.buffers())

    def preprocess(self, x):
        x = rearrange(x, "... d -> (...) d")
        return x

    def quantize(self, x):
        embed = self.embed.t()
        dist = -(
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = dist.max(dim=-1).indices
        return embed_ind

    def postprocess_emb(self, embed_ind, shape):
        return embed_ind.view(*shape[:-1])

    def dequantize(self, embed_ind):
        quantize = F.embedding(embed_ind, self.embed)
        return quantize

    def encode(self, x):
        shape = x.shape
        x = self.preprocess(x)
        embed_ind = self.quantize(x)
        embed_ind = self.postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind):
        quantize = self.dequantize(embed_ind)
        return quantize

    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        x = self.preprocess(x)

        self.init_embed_(x)

        embed_ind = self.quantize(x)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = self.postprocess_emb(embed_ind, shape)
        quantize = self.dequantize(embed_ind)

        if self.training:
            # We do the expiry of code at that point as buffers are in sync
            # and all the workers will take the same decision.
            self.expire_codes_(x)
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = x.t() @ embed_onehot
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size = (
                laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon)
                * self.cluster_size.sum()
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)

        return quantize, embed_ind


class VectorQuantization(nn.Module):
    """Vector quantization implementation.
    Currently supports only euclidean distance.
    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
        codebook_dim (int): Codebook dimension. If not defined, uses the specified dimension in dim.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
        commitment_weight (float): Weight for commitment loss.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim=None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
        commitment_weight: float = 1.0,
    ):
        super().__init__()
        _codebook_dim: int = default(codebook_dim, dim)

        requires_projection = _codebook_dim != dim
        self.project_in = (
            nn.Linear(dim, _codebook_dim) if requires_projection else nn.Identity()
        )
        self.project_out = (
            nn.Linear(_codebook_dim, dim) if requires_projection else nn.Identity()
        )

        self.epsilon = epsilon
        self.commitment_weight = commitment_weight

        self._codebook = EuclideanCodebook(
            dim=_codebook_dim,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            decay=decay,
            epsilon=epsilon,
            threshold_ema_dead_code=threshold_ema_dead_code,
        )
        self.codebook_size = codebook_size
        self.l2_loss = lambda a, b: (a - b) ** 2

    @property
    def codebook(self):
        return self._codebook.embed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        embed_in = self._codebook.encode(x)
        return embed_in

    def decode(self, embed_ind: torch.Tensor) -> torch.Tensor:
        quantize = self._codebook.decode(embed_ind)
        quantize = self.project_out(quantize)
        return quantize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: B x dim input tensor
        :return: quantize: B x dim tensor containing reconstruction after quantization
                 embed_ind: B-dimensional tensor containing embedding indices
                 loss: scalar tensor containing commitment loss
        """
        device = x.device
        x = self.project_in(x)

        quantize, embed_ind = self._codebook(x)

        if self.training:
            quantize = x + (quantize - x).detach()

        loss = torch.tensor([0.0], device=device, requires_grad=self.training)

        if self.training:
            if self.commitment_weight > 0:
                commit_loss = F.mse_loss(quantize.detach(), x)
                loss = loss + commit_loss * self.commitment_weight

        quantize = self.project_out(quantize)
        return quantize, embed_ind, loss


class ResidualVectorQuantization(nn.Module):
    """Residual vector quantization implementation.
    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, *, num_quantizers: int, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )

    def forward(self, x, B, T, mask, n_q=None):
        """
        :param x: B x dim tensor
        :return: quantized_out: B x dim tensor
                 out_indices: B x n_q LongTensor containing indices for each quantizer
                 out_losses: scalar tensor containing commitment loss
        """
        quantized_out = 0.0
        residual = x

        all_losses = []
        all_indices = []

        n_q = n_q or len(self.layers)

        for layer in self.layers[:n_q]:
            quantized, indices, loss = layer(residual)
            residual = (
                residual - quantized
            )  # would need quantizer.detach() to have commitment gradients beyond the first quantizer, but this seems to harm performance
            quantized_out = quantized_out + quantized

            all_indices.append(indices)
            all_losses.append(loss)

        out_indices = torch.stack(all_indices, dim=-1)
        out_losses = torch.mean(torch.stack(all_losses))
        return quantized_out, out_indices, out_losses

    def encode(self, x: torch.Tensor, n_q=None) -> torch.Tensor:
        """
        :param x: B x dim input tensor
        :return: B x n_q LongTensor containing indices for each quantizer
        """
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        for layer in self.layers[:n_q]:
            indices = layer.encode(residual)  # indices = 16 x 8 = B x T
            # print(indices.shape, residual.shape, x.shape)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = torch.stack(all_indices, dim=-1)
        return out_indices

    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        """
        :param q_indices: B x n_q LongTensor containing indices for each quantizer
        :return: B x dim tensor containing reconstruction after quantization
        """
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        q_indices = q_indices.permute(1, 0).contiguous()
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out


class TemporalVertexEncoder(nn.Module):
    def __init__(
        self,
        n_vertices: int = 338,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.input_dim = n_vertices
        self.enc = nn.Sequential(
            nn.Conv1d(self.input_dim, latent_dim, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(latent_dim, latent_dim, kernel_size=2, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(latent_dim, latent_dim, kernel_size=2, dilation=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(latent_dim, latent_dim, kernel_size=2, dilation=3),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(latent_dim, latent_dim, kernel_size=2, dilation=1),
        )
        self.receptive_field = 8

    def forward(self, verts):
        """
        :param verts: B x T x n_vertices x 3 tensor containing batched sequences of vertices
        :return: B x T x latent_dim tensor containing the latent representation
        """
        if verts.dim() == 4:
            verts = verts.permute(0, 2, 3, 1).contiguous()
            verts = verts.view(verts.shape[0], self.input_dim, verts.shape[3])
        else:
            verts = verts.permute(0, 2, 1)
        verts = nn.functional.pad(verts, pad=[self.receptive_field - 1, 0])
        x = self.enc(verts)
        x = x.permute(0, 2, 1).contiguous()
        return x


class TemporalVertexDecoder(nn.Module):
    def __init__(
        self,
        n_vertices: int = 338,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.output_dim = n_vertices
        self.project_mean_shape = nn.Linear(self.output_dim, latent_dim)
        self.dec = nn.Sequential(
            nn.Conv1d(latent_dim, latent_dim, kernel_size=2, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(latent_dim, latent_dim, kernel_size=2, dilation=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(latent_dim, latent_dim, kernel_size=2, dilation=3),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(latent_dim, latent_dim, kernel_size=2, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(latent_dim, self.output_dim, kernel_size=1),
        )
        self.receptive_field = 8

    def forward(self, x):
        """
        :param x: B x T x latent_dim tensor containing batched sequences of vertex encodings
        :return: B x T x n_vertices x 3 tensor containing batched sequences of vertices
        """
        x = x.permute(0, 2, 1).contiguous()
        x = nn.functional.pad(x, pad=[self.receptive_field - 1, 0])
        verts = self.dec(x)
        verts = verts.permute(0, 2, 1)
        return verts


class TemporalVertexCodec(nn.Module):
    def __init__(
        self,
        n_vertices: int = 338,
        latent_dim: int = 128,
        categories: int = 128,
        residual_depth: int = 4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.categories = categories
        self.residual_depth = residual_depth
        self.n_clusters = categories
        self.encoder = TemporalVertexEncoder(
            n_vertices=n_vertices, latent_dim=latent_dim
        )
        self.decoder = TemporalVertexDecoder(
            n_vertices=n_vertices, latent_dim=latent_dim
        )
        self.quantizer = ResidualVectorQuantization(
            dim=latent_dim,
            codebook_size=categories,
            num_quantizers=residual_depth,
            decay=0.99,
            kmeans_init=True,
            kmeans_iters=10,
            threshold_ema_dead_code=2,
        )

    def predict(self, verts):
        """wrapper to provide compatibility with kmeans"""
        return self.encode(verts)

    def encode(self, verts):
        """
        :param verts: B x T x n_vertices x 3 tensor containing batched sequences of vertices
        :return: B x T x categories x residual_depth LongTensor containing quantized encodings
        """
        enc = self.encoder(verts)
        q = self.quantizer.encode(enc)
        return q

    def decode(self, q):
        """
        :param q: B x T x categories x residual_depth LongTensor containing quantized encodings
        :return: B x T x n_vertices x 3 tensor containing decoded vertices
        """
        reformat = q.dim() > 2
        if reformat:
            B, T, _ = q.shape
            q = q.reshape((-1, self.residual_depth))
        enc = self.quantizer.decode(q)
        if reformat:
            enc = enc.reshape((B, T, -1))
        verts = self.decoder(enc)
        return verts

    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        # Calculate new centres
        code_onehot = torch.zeros(
            self.categories, code_idx.shape[0], device=code_idx.device
        )  # categories, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # categories
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    def forward(self, verts, mask=None):
        """
        :param verts: B x T x n_vertices x 3 tensor containing mesh sequences
        :return: verts: B x T x n_vertices x 3 tensor containing reconstructed mesh sequences
                 vq_loss: scalar tensor for vq commitment loss
        """
        B, T = verts.shape[0], verts.shape[1]
        x = self.encoder(verts)
        x, code_idx, vq_loss = self.quantizer(
            x.view(B * T, self.latent_dim), B, T, mask
        )
        perplexity = self.compute_perplexity(code_idx[:, -1].view((-1)))
        verts = self.decoder(x.view(B, T, self.latent_dim))
        verts = verts.reshape((verts.shape[0], verts.shape[1], -1))
        return verts, vq_loss, perplexity
