"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os

from typing import Dict, List

import numpy as np
import torch
import torchaudio
from data_loaders.data import Social
from data_loaders.tensors import social_collate
from torch.utils.data import DataLoader
from utils.misc import prGreen


def get_dataset_loader(
    args,
    data_dict: Dict[str, np.ndarray],
    split: str = "train",
    chunk: bool = False,
    add_padding: bool = True,
) -> DataLoader:
    dataset = Social(
        args=args,
        data_dict=data_dict,
        split=split,
        chunk=chunk,
        add_padding=add_padding,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=not split == "test",
        num_workers=8,
        drop_last=True,
        collate_fn=social_collate,
        pin_memory=True,
    )
    return loader


def _load_pose_data(
    all_paths: List[str], audio_per_frame: int, flip_person: bool = False
) -> Dict[str, List]:
    data = []
    face = []
    audio = []
    lengths = []
    missing = []
    for _, curr_path_name in enumerate(all_paths):
        if not curr_path_name.endswith("_body_pose.npy"):
            continue
        # load face information and deal with missing codes
        curr_code = np.load(
            curr_path_name.replace("_body_pose.npy", "_face_expression.npy")
        ).astype(float)
        # curr_code = np.array(curr_face["codes"], dtype=float)
        missing_list = np.load(
            curr_path_name.replace("_body_pose.npy", "_missing_face_frames.npy")
        )
        if len(missing_list) == len(curr_code):
            print("skipping", curr_path_name, curr_code.shape)
            continue
        curr_missing = np.ones_like(curr_code)
        curr_missing[missing_list] = 0.0

        # load pose information and deal with discontinuities
        curr_pose = np.load(curr_path_name)
        if "PXB184" in curr_path_name or "RLW104" in curr_path_name:  # Capture 1 or 2
            curr_pose[:, 3] = (curr_pose[:, 3] + np.pi) % (2 * np.pi)
            curr_pose[:, 3] = (curr_pose[:, 3] + np.pi) % (2 * np.pi)

        # load audio information
        curr_audio, _ = torchaudio.load(
            curr_path_name.replace("_body_pose.npy", "_audio.wav")
        )
        curr_audio = curr_audio.T
        if flip_person:
            prGreen("[get_data.py] flipping the dataset of left right person")
            tmp = torch.zeros_like(curr_audio)
            tmp[:, 1] = curr_audio[:, 0]
            tmp[:, 0] = curr_audio[:, 1]
            curr_audio = tmp

        assert len(curr_pose) * audio_per_frame == len(
            curr_audio
        ), f"motion {curr_pose.shape} vs audio {curr_audio.shape}"

        data.append(curr_pose)
        face.append(curr_code)
        missing.append(curr_missing)
        audio.append(curr_audio)
        lengths.append(len(curr_pose))

    data_dict = {
        "data": data,
        "face": face,
        "audio": audio,
        "lengths": lengths,
        "missing": missing,
    }
    return data_dict


def load_local_data(
    data_root: str, audio_per_frame: int, flip_person: bool = False
) -> Dict[str, List]:
    if flip_person:
        if "PXB184" in data_root:
            data_root = data_root.replace("PXB184", "RLW104")
        elif "RLW104" in data_root:
            data_root = data_root.replace("RLW104", "PXB184")
        elif "TXB805" in data_root:
            data_root = data_root.replace("TXB805", "GQS883")
        elif "GQS883" in data_root:
            data_root = data_root.replace("GQS883", "TXB805")

    all_paths = [os.path.join(data_root, x) for x in os.listdir(data_root)]
    all_paths.sort()
    return _load_pose_data(
        all_paths,
        audio_per_frame,
        flip_person=flip_person,
    )
