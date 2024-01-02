"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch as th
import os
import re
import glob
import copy
from typing import Dict, Any, Iterator, Mapping, Optional, Union, Tuple, List


from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf,  DictConfig

from torch.optim.lr_scheduler import LRScheduler

from visualize.ca_body.utils.torch import to_device
from visualize.ca_body.utils.module_loader import load_class, build_optimizer

import torch.nn as nn

import logging

logging.basicConfig(
    format="[%(asctime)s][%(levelname)s][%(name)s]:%(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def process_losses(
    loss_dict: Dict[str, Any], reduce: bool = True, detach: bool = True
) -> Dict[str, th.Tensor]:
    """Preprocess the dict of losses outputs."""
    result = {k.replace("loss_", ""): v for k, v in loss_dict.items() if k.startswith("loss_")}
    if detach:
        result = {k: v.detach() for k, v in result.items()}
    if reduce:
        result = {k: float(v.mean().item()) for k, v in result.items()}
    return result



def load_config(path: str) -> DictConfig:
    # NOTE: THIS IS THE ONLY PLACE WHERE WE MODIFY CONFIG
    config = OmegaConf.load(path)

    # TODO: we should need to get rid of this in favor of DB
    assert 'CARE_ROOT' in os.environ
    config.CARE_ROOT = os.environ['CARE_ROOT']
    logger.info(f'{config.CARE_ROOT=}')

    if not os.path.isabs(config.train.run_dir):
        config.train.run_dir = os.path.join(os.environ['CARE_ROOT'], config.train.run_dir)
    logger.info(f'{config.train.run_dir=}')
    os.makedirs(config.train.run_dir, exist_ok=True)
    return config


def load_from_config(config: Mapping[str, Any], **kwargs):
    """Instantiate an object given a config and arguments."""
    assert 'class_name' in config and 'module_name' not in config
    config = copy.deepcopy(config)
    ckpt = None if 'ckpt' not in config else config.pop('ckpt')
    class_name = config.pop('class_name')
    object_class = load_class(class_name)
    instance = object_class(**config, **kwargs)
    if ckpt is not None:
        load_checkpoint(
            ckpt_path=ckpt.path,
            modules={ckpt.get('module_name', 'model'): instance},
            ignore_names=ckpt.get('ignore_names', []),
            strict=ckpt.get('strict', False),
        )
    return instance


def save_checkpoint(ckpt_path, modules: Dict[str, Any], iteration=None, keep_last_k=None):
    if keep_last_k is not None:
        raise NotImplementedError()
    ckpt_dict = {}
    if os.path.isdir(ckpt_path):
        assert iteration is not None
        ckpt_path = os.path.join(ckpt_path, f"{iteration:06d}.pt")
        ckpt_dict["iteration"] = iteration
    for name, mod in modules.items():
        if hasattr(mod, "module"):
            mod = mod.module
        ckpt_dict[name] = mod.state_dict()
    th.save(ckpt_dict, ckpt_path)


def filter_params(params, ignore_names):
    return OrderedDict(
        [
            (k, v)
            for k, v in params.items()
            if not any([re.match(n, k) is not None for n in ignore_names])
        ]
    )


def save_file_summaries(path: str, summaries: Dict[str, Tuple[str, Any]]):
    """Saving regular summaries for monitoring purposes."""
    for name, (value, ext) in summaries.items():
        #save(f'{path}/{name}.{ext}', value)
        raise NotImplementedError()


def load_checkpoint(
    ckpt_path: str,
    modules: Dict[str, Any],
    iteration: int =None,
    strict: bool =False,
    map_location: Optional[str] =None,
    ignore_names: Optional[Dict[str, List[str]]]=None,
):
    """Load a checkpoint.
    Args:
        ckpt_path: directory or the full path to the checkpoint
    """
    if map_location is None:
        map_location = "cpu"
    # adding
    if os.path.isdir(ckpt_path):
        if iteration is None:
            # lookup latest iteration
            iteration = max(
                [
                    int(os.path.splitext(os.path.basename(p))[0])
                    for p in glob.glob(os.path.join(ckpt_path, "*.pt"))
                ]
            )
        ckpt_path = os.path.join(ckpt_path, f"{iteration:06d}.pt")
    logger.info(f"loading checkpoint {ckpt_path}")
    ckpt_dict = th.load(ckpt_path, map_location=map_location)
    for name, mod in modules.items():
        params = ckpt_dict[name]
        if ignore_names is not None and name in ignore_names:
            logger.info(f"skipping: {ignore_names[name]}")
            params = filter_params(params, ignore_names[name])
        mod.load_state_dict(params, strict=strict)


def train(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: th.optim.Optimizer,
    train_data: Iterator,
    config: Mapping[str, Any],
    lr_scheduler: Optional[LRScheduler] = None,
    train_writer: Optional[SummaryWriter] = None,
    saving_enabled: bool = True,
    logging_enabled: bool = True,
    iteration: int = 0,
    device: Optional[Union[th.device, str]] = "cuda:0",
) -> None:

    for batch in train_data:
        if batch is None:
            logger.info("skipping empty batch")
            continue
        batch = to_device(batch, device)
        batch["iteration"] = iteration

        # leaving only inputs acutally used by the model
        preds = model(**filter_inputs(batch, model, required_only=False))

        # TODO: switch to the old-school loss computation
        loss, loss_dict = loss_fn(preds, batch, iteration=iteration)
        assert not th.isnan(loss), "loss is NaN"

        if th.isnan(loss):
            _loss_dict = process_losses(loss_dict)
            loss_str = " ".join([f"{k}={v:.4f}" for k, v in _loss_dict.items()])
            logger.info(f"iter={iteration}: {loss_str}")
            raise ValueError("loss is NaN")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if logging_enabled and iteration % config.train.log_every_n_steps == 0:
            _loss_dict = process_losses(loss_dict)
            loss_str = " ".join([f"{k}={v:.4f}" for k, v in _loss_dict.items()])
            logger.info(f"iter={iteration}: {loss_str}")

        if logging_enabled and train_writer and iteration % config.train.log_every_n_steps == 0:
            for name, value in _loss_dict.items():
                train_writer.add_scalar(f"Losses/{name}", value, global_step=iteration)
            train_writer.flush()

        if saving_enabled and iteration % config.train.ckpt_every_n_steps == 0:
            logger.info(f"iter={iteration}: saving checkpoint to `{config.train.ckpt_dir}`")
            save_checkpoint(
                config.train.ckpt_dir,
                {"model": model, "optimizer": optimizer},
                iteration=iteration,
            )

        if logging_enabled and iteration % config.train.summary_every_n_steps == 0:
            summaries = model.compute_summaries(preds, batch)
            save_file_summaries(config.train.run_dir, summaries, prefix="train")

        if lr_scheduler is not None and iteration and iteration % config.train.update_lr_every == 0:
            lr_scheduler.step()

        iteration += 1
        if iteration >= config.train.n_max_iters:
            logger.info(f"reached max number of iters ({config.train.n_max_iters})")
            break

    if saving_enabled:
        logger.info(f"saving the final checkpoint to `{config.train.run_dir}/model.pt`")
        save_checkpoint(f"{config.train.run_dir}/model.pt", {"model": model})

