"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import cProfile as profile
import functools
import pstats

import blobfile as bf
import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

import utils.logger as logger
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import LossAwareSampler, create_named_schedule_sampler
from utils.misc import dev, load_state_dict

INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self, args, train_platform, model, diffusion, data, writer, rank=0, world_size=1
    ):
        self.args = args
        self.dataset = args.dataset
        self.train_platform = train_platform
        self.model = model
        self.diffusion = diffusion
        self.cond_mode = model.module.cond_mode if world_size > 1 else model.cond_mode
        self.data = data
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps
        self.rank = rank
        self.world_size = world_size

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1
        chunks = list(range(self.num_steps))
        num_chunks = int(self.num_steps / 10)
        chunks = np.array_split(chunks, num_chunks)
        self.chunks = np.reshape(chunks[10_000::10], (-1))
        self.sync_cuda = torch.cuda.is_available()
        self.writer = writer

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.rank}")

        self.schedule_sampler_type = "uniform"
        self.schedule_sampler = create_named_schedule_sampler(
            self.schedule_sampler_type, diffusion
        )
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None
        self.use_ddp = True
        self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                load_state_dict(resume_checkpoint, map_location=dev())
            )

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = load_state_dict(opt_checkpoint, map_location=dev())
            self.opt.load_state_dict(state_dict)

    def _print_stats(self, logger):
        if (self.step % 100 == 0 and self.step > 0) and self.rank == 0:
            v = logger.get_current().name2val
            v = v["loss"]
            print("step[{}]: loss[{:0.5f}]".format(self.step + self.resume_step, v))

    def _write_to_logger(self, logger):
        if (self.step % self.log_interval == 0) and self.rank == 0:
            for k, v in logger.get_current().name2val.items():
                if k == "loss":
                    print(
                        "step[{}]: loss[{:0.5f}]".format(
                            self.step + self.resume_step, v
                        )
                    )
                    self.writer.add_scalar(f"./Train/{k}", v, self.step)
                if k in ["step", "samples"] or "_q" in k:
                    continue
                else:
                    self.train_platform.report_scalar(
                        name=k, value=v, iteration=self.step, group_name="Loss"
                    )
                    self.writer.add_scalar(f"./Train/{k}", v, self.step)

    def run_loop(self):
        for _ in range(self.num_epochs):
            if self.rank == 0:
                prof = profile.Profile()
                prof.enable()

            for motion, cond in tqdm(self.data, disable=(self.rank != 0)):
                if not (
                    not self.lr_anneal_steps
                    or self.step + self.resume_step < self.lr_anneal_steps
                ):
                    break

                motion = motion.to(self.device)
                cond["y"] = {
                    key: val.to(self.device) if torch.is_tensor(val) else val
                    for key, val in cond["y"].items()
                }
                self.run_step(motion, cond)
                self._print_stats(logger)
                self._write_to_logger(logger)
                if (self.step % self.save_interval == 0) and self.rank == 0:
                    self.save()

                self.step += 1

                if (self.step == 1000) and self.rank == 0:
                    prof.disable()
                    stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
                    stats.print_stats(10)

            if not (
                not self.lr_anneal_steps
                or self.step + self.resume_step < self.lr_anneal_steps
            ):
                break

        # Save the last checkpoint if it wasn't already saved.
        if ((self.step - 1) % self.save_interval != 0) and self.rank == 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.mp_trainer.optimize(self.opt)
        self._anneal_lr()
        if self.rank == 0:
            self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], batch.device)

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"

    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)

            # Do not save CLIP weights
            clip_weights = [e for e in state_dict.keys() if e.startswith("clip_model.")]
            for e in clip_weights:
                del state_dict[e]

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.step+self.resume_step):09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
