"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import json
import os
from argparse import ArgumentParser


def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ["dataset", "model", "diffusion"]:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)
    args_to_overwrite += ["data_root"]

    # load args from model
    model_path = get_model_path_from_args()
    args_path = os.path.join(os.path.dirname(model_path), "args.json")
    print(args_path)
    assert os.path.exists(args_path), "Arguments json file was not found!"
    with open(args_path, "r") as fr:
        model_args = json.load(fr)

    for a in args_to_overwrite:
        if a in model_args.keys():
            if a == "timestep_respacing" or a == "partial":
                continue
            setattr(args, a, model_args[a])

        elif "cond_mode" in model_args:  # backward compitability
            unconstrained = model_args["cond_mode"] == "no_cond"
            setattr(args, "unconstrained", unconstrained)

        else:
            print(
                "Warning: was not able to load [{}], using default value [{}] instead.".format(
                    a, args.__dict__[a]
                )
            )

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    return args


def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {
                a.dest: getattr(args, a.dest, None) for a in group._group_actions
            }
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError("group_name was not found.")


def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument("model_path")
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        raise ValueError("model_path argument must be specified.")


def add_base_options(parser):
    group = parser.add_argument_group("base")
    group.add_argument(
        "--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU."
    )
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument(
        "--batch_size", default=64, type=int, help="Batch size during training."
    )


def add_diffusion_options(parser):
    group = parser.add_argument_group("diffusion")
    group.add_argument(
        "--noise_schedule",
        default="cosine",
        choices=["linear", "cosine"],
        type=str,
        help="Noise schedule type",
    )
    group.add_argument(
        "--diffusion_steps",
        default=10,
        type=int,
        help="Number of diffusion steps (denoted T in the paper)",
    )
    group.add_argument(
        "--timestep_respacing",
        default="ddim100",
        type=str,
        help="ddimN, else empty string",
    )
    group.add_argument(
        "--sigma_small", default=True, type=bool, help="Use smaller sigma values."
    )


def add_model_options(parser):
    group = parser.add_argument_group("model")
    group.add_argument("--layers", default=8, type=int, help="Number of layers.")
    group.add_argument(
        "--num_audio_layers", default=3, type=int, help="Number of audio layers."
    )
    group.add_argument("--heads", default=4, type=int, help="Number of heads.")
    group.add_argument(
        "--latent_dim", default=512, type=int, help="Transformer/GRU width."
    )
    group.add_argument(
        "--cond_mask_prob",
        default=0.20,
        type=float,
        help="The probability of masking the condition during training."
        " For classifier-free guidance learning.",
    )
    group.add_argument(
        "--lambda_vel", default=0.0, type=float, help="Joint velocity loss."
    )
    group.add_argument(
        "--unconstrained",
        action="store_true",
        help="Model is trained unconditionally. That is, it is constrained by neither text nor action. "
        "Currently tested on HumanAct12 only.",
    )
    group.add_argument(
        "--data_format",
        type=str,
        choices=["pose", "face"],
        default="pose",
        help="whether or not to use vae for diffusion process",
    )
    group.add_argument("--not_rotary", action="store_true")
    group.add_argument("--simplify_audio", action="store_true")
    group.add_argument("--add_frame_cond", type=float, choices=[1], default=None)


def add_data_options(parser):
    group = parser.add_argument_group("dataset")
    group.add_argument(
        "--dataset",
        default="social",
        choices=["social"],
        type=str,
        help="Dataset name (choose from list).",
    )
    group.add_argument("--data_root", type=str, default=None, help="dataset directory")
    group.add_argument("--max_seq_length", default=600, type=int)
    group.add_argument(
        "--split", type=str, default=None, choices=["test", "train", "val"]
    )


def add_training_options(parser):
    group = parser.add_argument_group("training")
    group.add_argument(
        "--save_dir",
        required=True,
        type=str,
        help="Path to save checkpoints and results.",
    )
    group.add_argument(
        "--overwrite",
        action="store_true",
        help="If True, will enable to use an already existing save_dir.",
    )
    group.add_argument(
        "--train_platform_type",
        default="NoPlatform",
        choices=["NoPlatform", "ClearmlPlatform", "TensorboardPlatform"],
        type=str,
        help="Choose platform to log results. NoPlatform means no logging.",
    )
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument(
        "--weight_decay", default=0.0, type=float, help="Optimizer weight decay."
    )
    group.add_argument(
        "--lr_anneal_steps",
        default=0,
        type=int,
        help="Number of learning rate anneal steps.",
    )
    group.add_argument(
        "--log_interval", default=1_000, type=int, help="Log losses each N steps"
    )
    group.add_argument(
        "--save_interval",
        default=5_000,
        type=int,
        help="Save checkpoints and run evaluation each N steps",
    )
    group.add_argument(
        "--num_steps",
        default=800_000,
        type=int,
        help="Training will stop after the specified number of steps.",
    )
    group.add_argument(
        "--resume_checkpoint",
        default="",
        type=str,
        help="If not empty, will start from the specified checkpoint (path to model###.pt file).",
    )


def add_sampling_options(parser):
    group = parser.add_argument_group("sampling")
    group.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="Path to model####.pt file to be sampled.",
    )
    group.add_argument(
        "--output_dir",
        default="",
        type=str,
        help="Path to results dir (auto created by the script). "
        "If empty, will create dir in parallel to checkpoint.",
    )
    group.add_argument("--face_codes", default=None, type=str)
    group.add_argument("--pose_codes", default=None, type=str)
    group.add_argument(
        "--num_samples",
        default=10,
        type=int,
        help="Maximal number of prompts to sample, "
        "if loading dataset from file, this field will be ignored.",
    )
    group.add_argument(
        "--num_repetitions",
        default=3,
        type=int,
        help="Number of repetitions, per sample (text prompt/action)",
    )
    group.add_argument(
        "--guidance_param",
        default=2.5,
        type=float,
        help="For classifier-free sampling - specifies the s parameter, as defined in the paper.",
    )
    group.add_argument(
        "--curr_seq_length",
        default=None,
        type=int,
    )
    group.add_argument(
        "--render_gt",
        action="store_true",
        help="whether to use pretrained clipmodel for audio encoding",
    )


def add_generate_options(parser):
    group = parser.add_argument_group("generate")
    group.add_argument(
        "--plot",
        action="store_true",
        help="Whether or not to save the renderings as a video.",
    )
    group.add_argument(
        "--resume_trans",
        default=None,
        type=str,
        help="keyframe prediction network.",
    )
    group.add_argument("--flip_person", action="store_true")


def get_cond_mode(args):
    if args.dataset == "social":
        cond_mode = "audio"
    return cond_mode


def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    return parser.parse_args()


def generate_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    args = parse_and_load_from_model(parser)
    return args
