"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import copy
import json
import logging
import os
import sys
import warnings
from typing import Any, Dict

import model.vqvae as vqvae

import numpy as np
import torch
import torch.optim as optim
from data_loaders.get_data import get_dataset_loader, load_local_data
from diffusion.nn import sum_flat
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.vq_parser_utils import train_args

warnings.filterwarnings("ignore")


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def get_logger(out_dir: str):
    logger = logging.getLogger("Exp")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_path = os.path.join(out_dir, "run.log")
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger


class ModelTrainer:
    def __init__(self, args, net: vqvae.TemporalVertexCodec, logger, writer):
        self.net = net
        self.warm_up_iter = args.warm_up_iter
        self.lr = args.lr
        self.optimizer = optim.AdamW(
            self.net.parameters(),
            lr=args.lr,
            betas=(0.9, 0.99),
            weight_decay=args.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=args.lr_scheduler, gamma=args.gamma
        )
        self.data_format = args.data_format
        self.loss = torch.nn.SmoothL1Loss()
        self.loss_vel = args.loss_vel
        self.commit = args.commit
        self.logger = logger
        self.writer = writer
        self.best_commit = float("inf")
        self.best_recons = float("inf")
        self.best_perplexity = float("inf")
        self.best_iter = 0
        self.out_dir = args.out_dir

    def _masked_l2(self, a, b, mask):
        loss = self._l2_loss(a, b)
        loss = sum_flat(loss * mask.float())
        n_entries = a.shape[1] * a.shape[2]
        non_zero_elements = sum_flat(mask) * n_entries
        mse_loss_val = loss / non_zero_elements
        return mse_loss_val

    def _l2_loss(self, motion_pred, motion_gt, mask=None):
        if mask is not None:
            return self._masked_l2(motion_pred, motion_gt, mask)
        else:
            return self.loss(motion_pred, motion_gt)

    def _vel_loss(self, motion_pred, motion_gt):
        model_results_vel = motion_pred[..., :-1] - motion_pred[..., 1:]
        model_targets_vel = motion_gt[..., :-1] - motion_gt[..., 1:]
        return self.loss(model_results_vel, model_targets_vel)

    def _update_lr_warm_up(self, nb_iter):
        current_lr = self.lr * (nb_iter + 1) / (self.warm_up_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr
        return current_lr

    def run_warmup_steps(self, train_loader_iter, skip_step, logger):
        avg_recons, avg_perplexity, avg_commit = 0.0, 0.0, 0.0
        for nb_iter in tqdm(range(1, args.warm_up_iter)):
            current_lr = self._update_lr_warm_up(nb_iter)
            gt_motion, cond = next(train_loader_iter)
            loss_dict = self.run_train_step(gt_motion, cond, skip_step)

            avg_recons += loss_dict["loss_motion"]
            avg_perplexity += loss_dict["perplexity"]
            avg_commit += loss_dict["loss_commit"]

            if nb_iter % args.print_iter == 0:
                avg_recons /= args.print_iter
                avg_perplexity /= args.print_iter
                avg_commit /= args.print_iter

                logger.info(
                    f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}"
                )

                avg_recons, avg_perplexity, avg_commit = 0.0, 0.0, 0.0

    def run_train_step(
        self, gt_motion: torch.Tensor, cond: torch.Tensor, skip_step: int
    ) -> Dict[str, Any]:
        self.net.train()
        loss_dict = {}
        # run model
        gt_motion = gt_motion.permute(0, 3, 1, 2).squeeze(-1).cuda().float()
        cond["y"] = {
            key: val.to(gt_motion.device) if torch.is_tensor(val) else val
            for key, val in cond["y"].items()
        }
        gt_motion = gt_motion[:, ::skip_step, :]
        pred_motion, loss_commit, perplexity = self.net(gt_motion, mask=None)
        loss_motion = self._l2_loss(pred_motion, gt_motion).mean()
        loss_vel = 0.0
        if self.loss_vel > 0:
            loss_vel = self._vel_loss(pred_motion, gt_motion)
        loss = loss_motion + self.commit * loss_commit + self.loss_vel * loss_vel
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # record losses
        if self.loss_vel > 0:
            loss_dict["vel"] = loss_vel.item()
        loss_dict["loss"] = loss.item()
        loss_dict["loss_motion"] = loss_motion.item()
        loss_dict["loss_commit"] = loss_commit.item()
        loss_dict["perplexity"] = perplexity.item()
        return loss_dict

    def save_model(self, save_path):
        torch.save(
            {
                "net": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler,
            },
            save_path,
        )

    def _save_predictions(self, name, unstd_pose, unstd_pred):
        curr_name = os.path.basename(name)
        path = os.path.join(self.out_dir, curr_name)
        for j in range(len(path.split("/")) - 1):
            if not os.path.exists("/".join(path.split("/")[: j + 1])):
                os.system("mkdir " + "/".join(path.split("/")[: j + 1]))
        np.save(os.path.join(self.out_dir, curr_name + "_gt.npy"), unstd_pose)
        np.save(os.path.join(self.out_dir, curr_name + "_pred.npy"), unstd_pred)

    def _log_losses(
        self,
        commit_loss: float,
        recons_loss: float,
        total_perplexity: float,
        nb_iter: int,
        nb_sample: int,
        draw: bool,
        save: bool,
    ) -> None:
        avg_commit = commit_loss / nb_sample
        avg_recons = recons_loss / nb_sample
        avg_perplexity = total_perplexity / nb_sample
        self.logger.info(
            f"Eval. Iter {nb_iter} : \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}"
        )

        if draw:
            self.writer.add_scalar("./Val/Perplexity", avg_perplexity, nb_iter)
            self.writer.add_scalar("./Val/Commit", avg_commit, nb_iter)
            self.writer.add_scalar("./Val/Recons", avg_recons, nb_iter)

        if avg_perplexity < self.best_perplexity:
            msg = f"--> --> \t Perplexity Improved from {self.best_perplexity:.5f} to {avg_perplexity:.5f} !!!"
            self.logger.info(msg)
            self.best_perplexity = avg_perplexity
            if save:
                print(f"saving checkpoint net_best.pth")
                self.save_model(os.path.join(self.out_dir, "net_best.pth"))

        if avg_commit < self.best_commit:
            msg = f"--> --> \t Commit Improved from {self.best_commit:.5f} to {avg_commit:.5f} !!!"
            self.logger.info(msg)
            self.best_commit = avg_commit

        if avg_recons < self.best_recons:
            msg = f"--> --> \t Recons Improved from {self.best_recons:.5f} to {avg_recons:.5f} !!!"
            self.logger.info(msg)
            self.best_recons = avg_recons

    @torch.no_grad()
    def evaluation_vqvae(
        self,
        val_loader,
        nb_iter: int,
        draw: bool = True,
        save: bool = True,
        savenpy: bool = False,
    ) -> None:
        self.net.eval()
        nb_sample = 0
        commit_loss = 0
        recons_loss = 0
        total_perplexity = 0
        for _, batch in enumerate(val_loader):
            motion, cond = batch
            m_length = cond["y"]["lengths"]
            motion = motion.permute(0, 3, 1, 2).squeeze(-1).cuda().float()
            cond["y"] = {
                key: val.to(motion.device) if torch.is_tensor(val) else val
                for key, val in cond["y"].items()
            }
            motion = motion[:, :: val_loader.dataset.step, :].cuda().float()
            bs, seq = motion.shape[0], motion.shape[1]
            pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()
            for i in range(bs):
                curr_gt = motion[i : i + 1, : m_length[i]]
                pred, loss_commit, perplexity = self.net(curr_gt)
                l2_loss = self._l2_loss(pred, curr_gt)
                recons_loss += l2_loss.mean().item()
                commit_loss += loss_commit
                total_perplexity += perplexity
                unstd_pred = val_loader.dataset.inv_transform(
                    pred.detach().cpu().numpy(), self.data_format
                )
                unstd_pose = val_loader.dataset.inv_transform(
                    curr_gt.detach().cpu().numpy(), self.data_format
                )
                if savenpy:
                    self._save_predictions(
                        "b{i:04d}", unstd_pose[:, : m_length[i]], unstd_pred
                    )
                pred_pose_eval[i : i + 1, : m_length[i], :] = pred
            nb_sample += bs

        self._log_losses(
            commit_loss, recons_loss, total_perplexity, nb_iter, nb_sample, draw, save
        )
        if save:
            print(f"saving checkpoint net_last.pth")
            self.save_model(os.path.join(self.out_dir, "net_last.pth"))
            if nb_iter % 100000 == 0:
                print(f"saving checkpoint net_iter_x.pth")
                self.save_model(
                    os.path.join(self.out_dir, "net_iter" + str(nb_iter) + ".pth")
                )


def _load_data_info(args, logger):
    data_dict = load_local_data(args.data_root, audio_per_frame=1600)
    train_loader = get_dataset_loader(
        args=args, data_dict=data_dict, split="train", add_padding=False
    )
    val_loader = get_dataset_loader(
        args=args, data_dict=data_dict, split="val", add_padding=False
    )

    logger.info(
        f"Training on {args.dataname}, motions are with {args.nb_joints} joints"
    )
    train_loader_iter = cycle(train_loader)
    skip_step = train_loader.dataset.step
    return train_loader_iter, val_loader, skip_step


def _load_checkpoint(args, net, logger):
    cp_dir = os.path.dirname(args.resume_pth)
    with open(f"{cp_dir}/args.json") as f:
        trans_args = json.load(f)
    assert trans_args["data_root"] == args.data_root, "data_root doesnt match"
    logger.info("loading checkpoint from {}".format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location="cpu")
    net.load_state_dict(ckpt["net"], strict=True)
    return net


def main(args):
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    logger = get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    if args.data_format == "pose":
        args.nb_joints = 104
    elif args.data_format == "face":
        args.nb_joints = 256

    args_path = os.path.join(args.out_dir, "args.json")
    with open(args_path, "w") as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    if not os.path.exists(args.data_root):
        args.data_root = args.data_root.replace("/home/", "/derived/")

    train_loader_iter, val_loader, skip_step = _load_data_info(args, logger)
    net = vqvae.TemporalVertexCodec(
        n_vertices=args.nb_joints,
        latent_dim=args.output_emb_width,
        categories=args.code_dim,
        residual_depth=args.depth,
    )
    if args.resume_pth:
        net = _load_checkpoint(args, net, logger)
    net.train()
    net.cuda()

    trainer = ModelTrainer(args, net, logger, writer)

    trainer.run_warmup_steps(train_loader_iter, skip_step, logger)
    avg_recons, avg_perplexity, avg_commit = 0.0, 0.0, 0.0
    with torch.no_grad():
        trainer.evaluation_vqvae(
            val_loader, 0, save=(args.total_iter > 0), savenpy=True
        )

    for nb_iter in range(1, args.total_iter + 1):
        gt_motion, cond = next(train_loader_iter)
        loss_dict = trainer.run_train_step(gt_motion, cond, skip_step)
        trainer.scheduler.step()

        avg_recons += loss_dict["loss_motion"]
        avg_perplexity += loss_dict["perplexity"]
        avg_commit += loss_dict["loss_commit"]

        if nb_iter % args.print_iter == 0:
            avg_recons /= args.print_iter
            avg_perplexity /= args.print_iter
            avg_commit /= args.print_iter

            writer.add_scalar("./Train/L1", avg_recons, nb_iter)
            writer.add_scalar("./Train/PPL", avg_perplexity, nb_iter)
            writer.add_scalar("./Train/Commit", avg_commit, nb_iter)

            logger.info(
                f"Train. Iter {nb_iter} : \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}"
            )

            avg_recons, avg_perplexity, avg_commit = (0.0, 0.0, 0.0)

        if nb_iter % args.eval_iter == 0:
            trainer.evaluation_vqvae(
                val_loader, nb_iter, save=(args.total_iter > 0), savenpy=True
            )


if __name__ == "__main__":
    args = train_args()
    main(args)
