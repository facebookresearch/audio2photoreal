"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os

import torch

from data_loaders.get_data import load_local_data

from tqdm import tqdm

from utils.diff_parser_utils import train_args
from utils.misc import fixseed
from utils.model_util import get_person_num
from visualize.render_codes import BodyRenderer


def main():
    args = train_args()
    fixseed(args.seed)
    args.num_repetitions = 1
    config_base = f"./checkpoints/ca_body/data/{get_person_num(args.data_root)}"
    body_renderer = BodyRenderer(
        config_base=config_base,
        render_rgb=True,
    ).to(args.device)
    data_root = args.data_root
    data_dict = load_local_data(data_root, audio_per_frame=1600)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    for i in range(len(data_dict["data"])):
        end_range = len(data_dict["data"][i]) - args.max_seq_length
        for chunk_idx in tqdm(range(0, end_range, args.max_seq_length)):
            chunk_end = chunk_idx + args.max_seq_length
            curr_data_chunk = data_dict["data"][i][chunk_idx:chunk_end, :]
            curr_face_chunk = data_dict["face"][i][chunk_idx:chunk_end, :]
            curr_audio_chunk = data_dict["audio"][i][
                chunk_idx * 1600 : chunk_end * 1600, :
            ].T
            render_data_block = {
                "audio": curr_audio_chunk,  # 2 x T
                "body_motion": curr_data_chunk,  # T x 104
                "face_motion": curr_face_chunk,  # T x 256
            }
            body_renderer.render_full_video(
                render_data_block,
                f"{args.save_dir}/scene{i}_{chunk_idx:04d}.mp4",
                audio_sr=48_000,
            )


if __name__ == "__main__":
    main()
