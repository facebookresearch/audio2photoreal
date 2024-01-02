"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch

from diffusion import gaussian_diffusion as gd
from diffusion.respace import space_timesteps, SpacedDiffusion
from model.diffusion import FiLMTransformer
from torch.nn import functional as F


def get_person_num(config_path):
    if "PXB184" in config_path:
        person = "PXB184"
    elif "RLW104" in config_path:
        person = "RLW104"
    elif "TXB805" in config_path:
        person = "TXB805"
    elif "GQS883" in config_path:
        person = "GQS883"
    else:
        assert False, f"something wrong with config: {config_path}"
    return person


def load_model(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0, unexpected_keys
    assert all(
        [
            k.startswith("transformer.") or k.startswith("tokenizer.")
            for k in missing_keys
        ]
    ), missing_keys


def create_model_and_diffusion(args, split_type):
    model = FiLMTransformer(**get_model_args(args, split_type=split_type)).to(
        torch.float32
    )
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion


def get_model_args(args, split_type):
    if args.data_format == "face":
        nfeat = 256
        lfeat = 512
    elif args.data_format == "pose":
        nfeat = 104
        lfeat = 256

    if not hasattr(args, "num_audio_layers"):
        args.num_audio_layers = 3  # backwards compat

    model_args = {
        "args": args,
        "nfeats": nfeat,
        "latent_dim": lfeat,
        "ff_size": 1024,
        "num_layers": args.layers,
        "num_heads": args.heads,
        "dropout": 0.1,
        "cond_feature_dim": 512 * 2,
        "activation": F.gelu,
        "use_rotary": not args.not_rotary,
        "cond_mode": "uncond" if args.unconstrained else "audio",
        "split_type": split_type,
        "num_audio_layers": args.num_audio_layers,
        "device": args.device,
    }
    return model_args


def create_gaussian_diffusion(args):
    predict_xstart = True
    steps = 1000
    scale_beta = 1.0
    timestep_respacing = args.timestep_respacing
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    name = args.save_dir if hasattr(args, "save_dir") else args.model_path
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        data_format=args.data_format,
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        model_path=name,
    )
