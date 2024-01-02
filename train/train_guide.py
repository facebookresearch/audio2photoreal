"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import json
import os
from typing import Any, Dict

import numpy as np
import torch
import torch.optim as optim

from data_loaders.get_data import get_dataset_loader, load_local_data
from diffusion.nn import sum_flat
from model.guide import GuideTransformer
from model.vqvae import setup_tokenizer, TemporalVertexCodec
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.guide_parser_utils import train_args
from utils.misc import fixseed


class ModelTrainer:
    def __init__(
        self, args, model: GuideTransformer, tokenizer: TemporalVertexCodec
    ) -> None:
        self.add_frame_cond = args.add_frame_cond
        self.data_format = args.data_format
        self.tokenizer = tokenizer
        self.model = model.cuda()
        self.gn = args.gn
        self.max_seq_length = args.max_seq_length
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.99),
            weight_decay=args.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=args.lr_scheduler, gamma=args.gamma
        )
        self.l2_loss = lambda a, b: (a - b) ** 2
        self.start_step = 0
        self.warm_up_iter = args.warm_up_iter
        self.lr = args.lr
        self.ce_loss = torch.nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.n_clusters + 1, label_smoothing=0.1
        )

        if args.resume_trans is not None:
            self._load_from_checkpoint()

    def _load_from_checkpoint(self) -> None:
        print("loading", args.resume_trans)
        ckpt = torch.load(args.resume_trans, map_location="cpu")
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.start_step = ckpt["iteration"]

    def _abbreviate(
        self, meshes: torch.Tensor, mask: torch.Tensor, step: int
    ) -> (torch.Tensor,):
        keyframes = meshes[..., ::step]
        new_mask = mask[..., ::step]
        return keyframes, new_mask

    def _prepare_tokens(
        self, meshes: torch.Tensor, mask: torch.Tensor
    ) -> (torch.Tensor,):
        if self.add_frame_cond == 1:
            keyframes, new_mask = self._abbreviate(meshes, mask, 30)
        elif self.add_frame_cond is None:
            keyframes, new_mask = self._abbreviate(meshes, mask, 1)

        meshes = keyframes.squeeze(2).permute((0, 2, 1))
        B, T, _ = meshes.shape
        target_tokens = self.tokenizer.predict(meshes)
        target_tokens = target_tokens.reshape(B, -1)
        input_tokens = torch.cat(
            [
                torch.zeros(
                    (B, 1), dtype=target_tokens.dtype, device=target_tokens.device
                )
                + self.model.tokens,
                target_tokens[:, :-1],
            ],
            axis=-1,
        )
        return input_tokens, target_tokens, new_mask, meshes.reshape((B, T, -1))

    def _run_single_train_step(self, input_tokens, audio, target_tokens):
        B, T = input_tokens.shape[0], input_tokens.shape[1]
        self.optimizer.zero_grad()
        logits = self.model(input_tokens, audio, cond_drop_prob=0.20)
        loss = self.ce_loss(
            logits.reshape((B * T, -1)), target_tokens.reshape((B * T)).long()
        )
        loss.backward()
        if self.gn:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        return logits, loss

    def _run_single_val_step(
        self, motion: torch.Tensor, cond: torch.Tensor
    ) -> Dict[str, Any]:
        self.model.eval()
        with torch.no_grad():
            motion = torch.as_tensor(motion).cuda()
            (
                input_tokens,
                target_tokens,
                new_mask,
                downsampled_gt,
            ) = self._prepare_tokens(motion, cond["mask"])
            audio = cond["audio"].cuda()

            new_mask = torch.as_tensor(new_mask)
            B, T = target_tokens.shape[0], target_tokens.shape[1]
            logits = self.model(input_tokens, audio)
            tokens = torch.argmax(logits, dim=-1).view(
                B, -1, self.tokenizer.residual_depth
            )
            pred = self.tokenizer.decode(tokens).detach().cpu()
            ce_loss = self.ce_loss(
                logits.reshape((B * T, -1)), target_tokens.reshape((B * T)).long()
            )
            l2_loss = self._masked_l2(
                downsampled_gt.permute(0, 2, 1).unsqueeze(2).detach().cpu(),
                pred.permute(0, 2, 1).unsqueeze(2),
                new_mask,
            )
            acc = self.compute_accuracy(logits, target_tokens, new_mask)

        return {
            "pred": pred,
            "gt": downsampled_gt,
            "metrics": {
                "ce_loss": ce_loss.item(),
                "l2_loss": l2_loss.item(),
                "perplexity": np.exp(ce_loss.item()),
                "acc": acc.item(),
            },
        }

    def _masked_l2(self, a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
        loss = self.l2_loss(a, b)
        loss = sum_flat(loss * mask.float())
        n_entries = a.shape[1] * a.shape[2]
        non_zero_elements = sum_flat(mask) * n_entries
        mse_loss_val = loss / non_zero_elements
        return mse_loss_val.mean()

    def compute_ce_loss(
        self, logits: torch.Tensor, target_tokens: torch.Tensor, mask: torch.Tensor
    ) -> float:
        target_tokens[~mask.squeeze().detach().cpu()] = 0
        B = logits.shape[0]
        logprobs = torch.log_softmax(logits, dim=-1).view(
            B, -1, 1, self.tokenizer.n_clusters
        )
        logprobs = logprobs[:, self.mask_left :, :, :].contiguous()
        labels = target_tokens.view(B, -1, 1)
        labels = labels[:, self.mask_left :, :].contiguous()
        loss = torch.nn.functional.nll_loss(
            logprobs.view(-1, self.tokenizer.n_clusters),
            labels.view(-1).long(),
            reduction="none",
        ).reshape((B, 1, 1, -1))
        mask = mask.float().to(loss.device)
        loss = sum_flat(loss * mask)
        non_zero_elements = sum_flat(mask)
        ce_loss_val = loss / non_zero_elements
        return ce_loss_val.mean()

    def compute_accuracy(
        self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> float:
        mask = mask.squeeze()
        probs = torch.softmax(logits, dim=-1)
        _, cls_pred_index = torch.max(probs, dim=-1)
        acc = (cls_pred_index.flatten(0) == target.flatten(0)).reshape(
            cls_pred_index.shape
        )
        acc = sum_flat(acc).detach().cpu()
        non_zero_elements = sum_flat(mask)
        acc_val = acc / non_zero_elements * 100
        return acc_val.mean()

    def update_lr_warm_up(self, nb_iter: int) -> float:
        current_lr = self.lr * (nb_iter + 1) / (self.warm_up_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr
        return current_lr

    def train_step(self, motion: torch.Tensor, cond: torch.Tensor) -> Dict[str, Any]:
        self.model.train()
        motion = torch.as_tensor(motion).cuda()
        input_tokens, target_tokens, new_mask, downsampled_gt = self._prepare_tokens(
            motion, cond["mask"]
        )
        audio = cond["audio"].cuda()
        new_mask = torch.as_tensor(new_mask)

        logits, loss = self._run_single_train_step(input_tokens, audio, target_tokens)
        with torch.no_grad():
            tokens = torch.argmax(logits, dim=-1).view(
                input_tokens.shape[0], -1, self.tokenizer.residual_depth
            )
            pred = self.tokenizer.decode(tokens).detach().cpu()
            l2_loss = self._masked_l2(
                downsampled_gt.permute(0, 2, 1).unsqueeze(2).detach().cpu(),
                pred.permute(0, 2, 1).unsqueeze(2),
                new_mask,
            )
            acc = self.compute_accuracy(logits, target_tokens, new_mask)

        return {
            "pred": pred,
            "gt": downsampled_gt,
            "loss": loss,
            "metrics": {
                "ce_loss": loss.item(),
                "l2_loss": l2_loss.item(),
                "perplexity": np.exp(loss.item()),
                "acc": acc.item(),
            },
        }

    def validate(
        self,
        val_data: DataLoader,
        writer: SummaryWriter,
        step: int,
        save_dir: str,
        log_step: int = 100,
        max_samples: int = 30,
    ) -> None:
        val_metrics = {}
        pred_values = []
        gt_values = []
        for i, (val_motion, val_cond) in enumerate(val_data):
            val_out = self._run_single_val_step(val_motion, val_cond["y"])
            if "metrics" in val_out.keys():
                for k, v in val_out["metrics"].items():
                    val_metrics[k] = val_metrics.get(k, 0.0) + v
            if "pred" in val_out.keys() and i % log_step == 0:
                pred_values.append(
                    val_data.dataset.inv_transform(val_out["pred"], self.data_format)
                )
                gt_values.append(
                    val_data.dataset.inv_transform(val_out["gt"], self.data_format)
                )
            if i % log_step == 0:
                print(
                    f'val_l2_loss at {step} [{i}]: {val_metrics["l2_loss"] / len(val_data):.4f}'
                )
        pred_values = torch.concatenate((pred_values), dim=0)
        gt_values = torch.concatenate((gt_values), dim=0)
        idx = np.random.permutation(len(pred_values))[:max_samples]
        pred_values = pred_values[idx]
        gt_values = gt_values[idx]
        for i, (pred, gt) in enumerate(zip(pred_values, gt_values)):
            pred = pred.unsqueeze(0).detach().cpu().numpy()
            pose = gt.unsqueeze(0).detach().cpu().numpy()
            np.save(os.path.join(save_dir, f"b{i:04d}_pred.npy"), pred)
            np.save(os.path.join(save_dir, f"b{i:04d}_gt.npy"), pose)

        msg = ""
        for k, v in val_metrics.items():
            writer.add_scalar(f"val_{k}", v / len(val_data), step)
            msg += f"val_{k} at {step}: {v / len(val_data):.4f} | "
        print(msg)


def _save_checkpoint(
    args, iteration: int, model: GuideTransformer, optimizer: optim.Optimizer
) -> None:
    os.makedirs(f"{args.out_dir}/checkpoints/", exist_ok=True)
    filename = f"iter-{iteration:07d}.pt"
    torch.save(
        {
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        f"{args.out_dir}/checkpoints/{filename}",
    )


def _load_data_info(args) -> (DataLoader, DataLoader):
    data_dict = load_local_data(args.data_root, audio_per_frame=1600)
    train_data = get_dataset_loader(
        args=args, data_dict=data_dict, split="train", add_padding=False
    )
    val_data = get_dataset_loader(args=args, data_dict=data_dict, split="val")
    return train_data, val_data


def main(args):
    fixseed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    writer = SummaryWriter(f"{args.out_dir}/logs/")
    args_path = os.path.join(args.out_dir, "args.json")
    with open(args_path, "w") as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)
    tokenizer = setup_tokenizer(args.resume_pth)

    model = GuideTransformer(
        tokens=tokenizer.n_clusters,
        emb_len=798 if args.max_seq_length == 240 else 1998,
        num_layers=args.layers,
        dim=args.dim,
    )
    train_data, val_data = _load_data_info(args)
    trainer = ModelTrainer(args, model, tokenizer)
    step = trainer.start_step

    for _ in range(1, args.total_iter + 1):
        train_metrics = {}
        count = 0
        for motion, cond in tqdm(train_data):
            if step < args.warm_up_iter:
                current_lr = trainer.update_lr_warm_up(step)

            # rum single train step
            train_out = trainer.train_step(motion, cond["y"])
            if "metrics" in train_out.keys():
                for k, v in train_out["metrics"].items():
                    train_metrics[k] = train_metrics.get(k, 0.0) + v
                count += 1

            # log all of the metrics
            if step % args.log_interval == 0:
                msg = ""
                for k, v in train_metrics.items():
                    writer.add_scalar(f"train_{k}", v / count, step)
                    msg += f"train_{k} at {step}: {v / count:.4f} | "
                    train_metrics = {}
                count = 0
                writer.add_scalar(f"train_lr", trainer.scheduler.get_lr()[0], step)
                if step < args.warm_up_iter:
                    msg += f"lr: {current_lr} | "
                print(msg)
                writer.flush()

            # run single evaluation step and save
            if step % args.eval_interval == 0:
                trainer.validate(val_data, writer, step, args.out_dir)
            if step % args.save_interval == 0:
                _save_checkpoint(args, step, trainer.model, trainer.optimizer)
            step += 1


if __name__ == "__main__":
    args = train_args()
    main(args)
