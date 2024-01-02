"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse


def _add_dataset_args(parser):
    parser.add_argument("--dataname", type=str, default="kit", help="dataset directory")
    parser.add_argument("--data_root", type=str, default=None, help="dataset directory")
    parser.add_argument("--max_seq_length", default=600, type=int)
    parser.add_argument("--add_frame_cond", type=float, choices=[1], default=None)
    parser.add_argument(
        "--data_format", type=str, default="pose", choices=["pose", "face"]
    )
    parser.add_argument("--dataset", default="social", type=str)
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")


def _add_optim_args(parser):
    parser.add_argument(
        "--total_iter",
        default=300_000,
        type=int,
        help="number of total iterations to run",
    )
    parser.add_argument(
        "--warm_up_iter",
        default=1000,
        type=int,
        help="number of total iterations for warmup",
    )
    parser.add_argument("--lr", default=2e-4, type=float, help="max learning rate")
    parser.add_argument(
        "--lr_scheduler",
        default=[300_000],
        nargs="+",
        type=int,
        help="learning rate schedule (iterations)",
    )
    parser.add_argument("--gamma", default=0.05, type=float, help="learning rate decay")

    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight decay")
    parser.add_argument(
        "--commit",
        type=float,
        default=0.02,
        help="hyper-parameter for the commitment loss",
    )
    parser.add_argument(
        "--loss_vel",
        type=float,
        default=0.1,
        help="hyper-parameter for the velocity loss",
    )


def _add_model_args(parser):
    parser.add_argument("--code_dim", type=int, default=512, help="embedding dimension")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument(
        "--output_emb_width", type=int, default=512, help="output embedding width"
    )
    parser.add_argument(
        "--resume_pth", type=str, default=None, help="resume pth for VQ"
    )


def train_args():
    parser = argparse.ArgumentParser(
        description="Optimal Transport AutoEncoder training for AIST",
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_dataset_args(parser)
    _add_optim_args(parser)
    _add_model_args(parser)

    ## output directory
    parser.add_argument("--out_dir", type=str, required=True, help="output directory")
    ## other
    parser.add_argument("--print_iter", default=200, type=int, help="print frequency")
    parser.add_argument(
        "--eval_iter", default=1000, type=int, help="evaluation frequency"
    )
    parser.add_argument(
        "--seed", default=123, type=int, help="seed for initializing training."
    )
    args = parser.parse_args()
    return args
