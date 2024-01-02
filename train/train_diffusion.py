"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import json
import os

import torch
import torch.multiprocessing as mp

from data_loaders.get_data import get_dataset_loader, load_local_data
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from train.train_platforms import ClearmlPlatform, NoPlatform, TensorboardPlatform
from train.training_loop import TrainLoop
from utils.diff_parser_utils import train_args
from utils.misc import cleanup, fixseed, setup_dist
from utils.model_util import create_model_and_diffusion


def main(rank: int, world_size: int):
    args = train_args()
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name="Args")
    setup_dist(args.device)

    if rank == 0:
        if args.save_dir is None:
            raise FileNotFoundError("save_dir was not specified.")
        elif os.path.exists(args.save_dir) and not args.overwrite:
            raise FileExistsError("save_dir [{}] already exists.".format(args.save_dir))
        elif not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        args_path = os.path.join(args.save_dir, "args.json")
        with open(args_path, "w") as fw:
            json.dump(vars(args), fw, indent=4, sort_keys=True)

    if not os.path.exists(args.data_root):
        args.data_root = args.data_root.replace("/home/", "/derived/")

    data_dict = load_local_data(args.data_root, audio_per_frame=1600)
    print("creating data loader...")
    data = get_dataset_loader(args=args, data_dict=data_dict)

    print("creating logger...")
    writer = SummaryWriter(args.save_dir)

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, split_type="train")
    model.to(rank)

    if world_size > 1:
        model = DDP(
            model, device_ids=[rank], output_device=rank, find_unused_parameters=True
        )

    params = (
        model.module.parameters_w_grad()
        if world_size > 1
        else model.parameters_w_grad()
    )
    print("Total params: %.2fM" % (sum(p.numel() for p in params) / 1000000.0))
    print("Training...")

    TrainLoop(
        args, train_platform, model, diffusion, data, writer, rank, world_size
    ).run_loop()
    train_platform.close()
    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"using {world_size} gpus")
    if world_size > 1:
        mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    else:
        main(rank=0, world_size=1)
