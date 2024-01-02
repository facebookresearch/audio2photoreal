"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os

from typing import Callable, Dict, Union

import numpy as np
import torch
from data_loaders.get_data import get_dataset_loader, load_local_data
from diffusion.respace import SpacedDiffusion
from model.cfg_sampler import ClassifierFreeSampleModel
from model.diffusion import FiLMTransformer

from torch.utils.data import DataLoader
from utils.diff_parser_utils import generate_args
from utils.misc import fixseed, prGreen
from utils.model_util import create_model_and_diffusion, get_person_num, load_model


def _construct_template_variables(unconstrained: bool) -> (str,):
    row_file_template = "sample{:02d}.mp4"
    all_file_template = "samples_{:02d}_to_{:02d}.mp4"
    if unconstrained:
        sample_file_template = "row{:02d}_col{:02d}.mp4"
        sample_print_template = "[{} row #{:02d} column #{:02d} | -> {}]"
        row_file_template = row_file_template.replace("sample", "row")
        row_print_template = "[{} row #{:02d} | all columns | -> {}]"
        all_file_template = all_file_template.replace("samples", "rows")
        all_print_template = "[rows {:02d} to {:02d} | -> {}]"
    else:
        sample_file_template = "sample{:02d}_rep{:02d}.mp4"
        sample_print_template = '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
        row_print_template = '[ "{}" ({:02d}) | all repetitions | -> {}]'
        all_print_template = "[samples {:02d} to {:02d} | all repetitions | -> {}]"

    return (
        sample_print_template,
        row_print_template,
        all_print_template,
        sample_file_template,
        row_file_template,
        all_file_template,
    )


def _replace_keyframes(
    model_kwargs: Dict[str, Dict[str, torch.Tensor]],
    model: Union[FiLMTransformer, ClassifierFreeSampleModel],
) -> torch.Tensor:
    B, T = (
        model_kwargs["y"]["keyframes"].shape[0],
        model_kwargs["y"]["keyframes"].shape[1],
    )
    with torch.no_grad():
        tokens = model.transformer.generate(
            model_kwargs["y"]["audio"],
            T,
            layers=model.tokenizer.residual_depth,
            n_sequences=B,
        )
    tokens = tokens.reshape((B, -1, model.tokenizer.residual_depth))
    pred = model.tokenizer.decode(tokens).detach().cpu()
    assert (
        model_kwargs["y"]["keyframes"].shape == pred.shape
    ), f"{model_kwargs['y']['keyframes'].shape} vs {pred.shape}"
    return pred


def _run_single_diffusion(
    args,
    model_kwargs: Dict[str, Dict[str, torch.Tensor]],
    diffusion: SpacedDiffusion,
    model: Union[FiLMTransformer, ClassifierFreeSampleModel],
    inv_transform: Callable,
    gt: torch.Tensor,
) -> (torch.Tensor,):
    if args.data_format == "pose" and args.resume_trans is not None:
        model_kwargs["y"]["keyframes"] = _replace_keyframes(model_kwargs, model)

    sample_fn = diffusion.ddim_sample_loop
    with torch.no_grad():
        sample = sample_fn(
            model,
            (args.batch_size, model.nfeats, 1, args.curr_seq_length),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
    sample = inv_transform(sample.cpu().permute(0, 2, 3, 1), args.data_format).permute(
        0, 3, 1, 2
    )
    curr_audio = inv_transform(model_kwargs["y"]["audio"].cpu().numpy(), "audio")
    keyframes = inv_transform(model_kwargs["y"]["keyframes"], args.data_format)
    gt_seq = inv_transform(gt.cpu().permute(0, 2, 3, 1), args.data_format).permute(
        0, 3, 1, 2
    )

    return sample, curr_audio, keyframes, gt_seq


def _generate_sequences(
    args,
    model_kwargs: Dict[str, Dict[str, torch.Tensor]],
    diffusion: SpacedDiffusion,
    model: Union[FiLMTransformer, ClassifierFreeSampleModel],
    test_data: torch.Tensor,
    gt: torch.Tensor,
) -> Dict[str, np.ndarray]:
    all_motions = []
    all_lengths = []
    all_audio = []
    all_gt = []
    all_keyframes = []

    for rep_i in range(args.num_repetitions):
        print(f"### Sampling [repetitions #{rep_i}]")
        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs["y"]["scale"] = (
                torch.ones(args.batch_size, device=args.device) * args.guidance_param
            )
        model_kwargs["y"] = {
            key: val.to(args.device) if torch.is_tensor(val) else val
            for key, val in model_kwargs["y"].items()
        }
        sample, curr_audio, keyframes, gt_seq = _run_single_diffusion(
            args, model_kwargs, diffusion, model, test_data.dataset.inv_transform, gt
        )
        all_motions.append(sample.cpu().numpy())
        all_audio.append(curr_audio)
        all_keyframes.append(keyframes.cpu().numpy())
        all_gt.append(gt_seq.cpu().numpy())
        all_lengths.append(model_kwargs["y"]["lengths"].cpu().numpy())

        print(f"created {len(all_motions) * args.batch_size} samples")

    return {
        "motions": np.concatenate(all_motions, axis=0),
        "audio": np.concatenate(all_audio, axis=0),
        "gt": np.concatenate(all_gt, axis=0),
        "lengths": np.concatenate(all_lengths, axis=0),
        "keyframes": np.concatenate(all_keyframes, axis=0),
    }


def _render_pred(
    args,
    data_block: Dict[str, torch.Tensor],
    sample_file_template: str,
    audio_per_frame: int,
) -> None:
    from visualize.render_codes import BodyRenderer

    face_codes = None
    if args.face_codes is not None:
        face_codes = np.load(args.face_codes, allow_pickle=True).item()
        face_motions = face_codes["motions"]
        face_gts = face_codes["gt"]
        face_audio = face_codes["audio"]

    config_base = f"./checkpoints/ca_body/data/{get_person_num(args.data_root)}"
    body_renderer = BodyRenderer(
        config_base=config_base,
        render_rgb=True,
    )

    for sample_i in range(args.num_samples):
        for rep_i in range(args.num_repetitions):
            idx = rep_i * args.batch_size + sample_i
            save_file = sample_file_template.format(sample_i, rep_i)
            animation_save_path = os.path.join(args.output_dir, save_file)
            # format data
            length = data_block["lengths"][idx]
            body_motion = (
                data_block["motions"][idx].transpose(2, 0, 1)[:length].squeeze(-1)
            )
            face_motion = face_motions[idx].transpose(2, 0, 1)[:length].squeeze(-1)
            assert np.array_equal(
                data_block["audio"][idx], face_audio[idx]
            ), "face audio is not the same"
            audio = data_block["audio"][idx, : length * audio_per_frame, :].T
            # set up render data block to pass into renderer
            render_data_block = {
                "audio": audio,
                "body_motion": body_motion,
                "face_motion": face_motion,
            }
            if args.render_gt:
                gt_body = data_block["gt"][idx].transpose(2, 0, 1)[:length].squeeze(-1)
                gt_face = face_gts[idx].transpose(2, 0, 1)[:length].squeeze(-1)
                render_data_block["gt_body"] = gt_body
                render_data_block["gt_face"] = gt_face
            body_renderer.render_full_video(
                render_data_block,
                animation_save_path,
                audio_sr=audio_per_frame * 30,
                render_gt=args.render_gt,
            )


def _reset_sample_args(args) -> None:
    # set the sequence length to match the one specified by user
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace("model", "").replace(".pt", "")
    args.curr_seq_length = (
        args.curr_seq_length
        if args.curr_seq_length is not None
        else args.max_seq_length
    )
    # add the resume predictor model path
    resume_trans_name = ""
    if args.data_format == "pose" and args.resume_trans is not None:
        resume_trans_parts = args.resume_trans.split("/")
        resume_trans_name = f"{resume_trans_parts[1]}_{resume_trans_parts[-1]}"
    # reformat the output directory
    args.output_dir = os.path.join(
        os.path.dirname(args.model_path),
        "samples_{}_{}_seed{}_{}".format(name, niter, args.seed, resume_trans_name),
    )
    assert (
        args.num_samples <= args.batch_size
    ), f"Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})"
    # set the batch size to match the number of samples to generate
    args.batch_size = args.num_samples


def _setup_dataset(args) -> DataLoader:
    data_root = args.data_root
    data_dict = load_local_data(
        data_root,
        audio_per_frame=1600,
        flip_person=args.flip_person,
    )
    test_data = get_dataset_loader(
        args=args,
        data_dict=data_dict,
        split="test",
        chunk=True,
    )
    return test_data


def _setup_model(
    args,
) -> (Union[FiLMTransformer, ClassifierFreeSampleModel], SpacedDiffusion):
    model, diffusion = create_model_and_diffusion(args, split_type="test")
    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location="cpu")
    load_model(model, state_dict)

    if not args.unconstrained:
        assert args.guidance_param != 1

    if args.guidance_param != 1:
        prGreen("[CFS] wrapping model in classifier free sample")
        model = ClassifierFreeSampleModel(model)
    model.to(args.device)
    model.eval()
    return model, diffusion


def main():
    args = generate_args()
    fixseed(args.seed)
    _reset_sample_args(args)

    print("Loading dataset...")
    test_data = _setup_dataset(args)
    iterator = iter(test_data)

    print("Creating model and diffusion...")
    model, diffusion = _setup_model(args)

    if args.pose_codes is None:
        # generate sequences
        gt, model_kwargs = next(iterator)
        data_block = _generate_sequences(
            args, model_kwargs, diffusion, model, test_data, gt
        )
        os.makedirs(args.output_dir, exist_ok=True)
        npy_path = os.path.join(args.output_dir, "results.npy")
        print(f"saving results file to [{npy_path}]")
        np.save(npy_path, data_block)
    else:
        # load the pre generated results
        data_block = np.load(args.pose_codes, allow_pickle=True).item()

    # plot function only if face_codes exist and we are on pose prediction
    if args.plot:
        assert args.face_codes is not None, "need body and faces"
        assert (
            args.data_format == "pose"
        ), "currently only supporting plot on pose stuff"
        print(f"saving visualizations to [{args.output_dir}]...")
        _, _, _, sample_file_template, _, _ = _construct_template_variables(
            args.unconstrained
        )
        _render_pred(
            args,
            data_block,
            sample_file_template,
            test_data.dataset.audio_per_frame,
        )


if __name__ == "__main__":
    main()
