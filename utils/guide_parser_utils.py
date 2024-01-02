"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse


def _add_dataset_args(parser):
    parser.add_argument("--dataset", default="social", type=str)
    parser.add_argument(
        "--data_format", type=str, default="pose", choices=["pose", "face"]
    )
    parser.add_argument("--data_root", type=str, default=None, help="dataset directory")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--add_frame_cond", type=int, default=None, choices=[1])
    parser.add_argument("--max_seq_length", default=600, type=int)


def _add_opt_args(parser):
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warm_up_iter", type=int, default=5_000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument(
        "--lr-scheduler",
        default=[50000, 400000],
        nargs="+",
        type=int,
        help="learning rate schedule (iterations)",
    )
    parser.add_argument("--gamma", default=0.1, type=float)
    parser.add_argument("--gn", action="store_true", help="gradient clipping")


def _add_model_args(parser):
    parser.add_argument("--layers", default=8, type=int)
    parser.add_argument("--dim", default=8, type=int)
    parser.add_argument("--resume_pth", type=str, required=True)
    parser.add_argument("--resume_trans", type=str, default=None)


def train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--total_iter", default=1_000_000, type=int)
    parser.add_argument("--log_interval", default=1_000, type=int)
    parser.add_argument("--eval_interval", default=1_000, type=int)
    parser.add_argument("--save_interval", default=5_000, type=int)
    _add_model_args(parser)
    _add_opt_args(parser)
    _add_dataset_args(parser)
    args = parser.parse_args()
    return args
