# From Audio to Photoreal Embodiment: Synthesizing Humans in Conversations
This repository contains a pytorch implementation of ["From Audio to Photoreal Embodiment: Synthesizing Humans in Conversations"](https://people.eecs.berkeley.edu/~evonne_ng/projects/audio2photoreal/)

:hatching_chick: **Try out our demo [here](https://colab.research.google.com/drive/1lnX3d-3T3LaO3nlN6R8s6pPvVNAk5mdK?usp=sharing)** or continue following the steps below to run code locally!
And thanks everyone for the support via contributions/comments/issues!

https://github.com/facebookresearch/audio2photoreal/assets/17986358/5cba4079-275e-48b6-aecc-f84f3108c810

This codebase provides:
- train code
- test code
- pretrained motion models
- access to dataset

If you use the dataset or code, please cite our [Paper](https://arxiv.org/abs/2401.01885)

```
@article{ng2024audio2photoreal,
  title={From Audio to Photoreal Embodiment: Synthesizing Humans in Conversations},
  author={Ng, Evonne and Romero, Javier and Bagautdinov, Timur and Bai, Shaojie and Darrell, Trevor and Kanazawa, Angjoo and Richard, Alexander},
  journal={arXiv preprint arXiv:2401.01885},
  year={2024}
}
```

### Repository Contents

- [**Quickstart:**](#quickstart) easy gradio demo that lets you record audio and render a video
- [**Installation:**](#installation) environment setup and installation (for more details on the rendering pipeline, please refer to [Codec Avatar Body](https://github.com/facebookresearch/ca_body))
- [**Download data and models:**](#download-data-and-models) download annotations and pre-trained models
    - [Dataset desc.](#dataset): description of dataset annotations
    - [Visualize Dataset](#visualize-ground-truth): script for visualizing ground truth annotations
    - [model desc.](#pretrained-models): description of pretrained models
- [**Running the pretrained models:**](#running-the-pretrained-models) how to generate results files and visualize the results using the rendering pipeline.
    - [Face generation](#face-generation): commands to generate the results file for the faces
    - [Body generation](#body-generation): commands to generate the results file for the bodies
    - [Visualization](#visualization): how to call into the rendering api. For full details, please refer to [this repo](https://github.com/facebookresearch/ca_body).
- [**Training from scratch (3 models):**](#training-from-scratch) scripts to get the training pipeline running from scratch for face, guide poses, and body models.
    - [Face diffusion model](#1-face-diffusion-model)
    - [Body diffusion](#2-body-diffusion-model)
    - [Body vq vae](#3-body-vq-vae)
    - [Body guide transformer](#4-body-guide-transformer)

We annotate code that you can directly copy and paste into your terminal using the :point_down: icon.

# Quickstart
With this demo, you can record an audio clip and select the number of samples you want to generate. 

Make sure you have CUDA 11.7 and gcc/++ 9.0 for pytorch3d compatibility

:point_down: Install necessary components. This will do the environment configuration and install the corresponding rendering assets, prerequisite models, and pretrained models:
```
conda create --name a2p_env python=3.9
conda activate a2p_env
sh demo/install.sh
```
:point_down: Run the demo. You can record your audio and then render corresponding results!
```
python -m demo.demo
```

:microphone: First, record your audio

![](assets/demo1.gif)

:hourglass: Hold tight because the rendering can take a while! 

You can change the number of samples (1-10) you want to generate, and download your favorite video by clicking on the download button on the top right of each video.

![](assets/demo2.gif)

# Installation
The code has been tested with CUDA 11.7 and python 3.9, gcc/++ 9.0

:point_down: If you haven't done so already via the demo setup, configure the environments and download prerequisite models:
```
conda create --name a2p_env python=3.9
conda activate a2p_env
pip install -r scripts/requirements.txt
sh scripts/download_prereq.sh
```
:point_down: To get the rendering working, please also make sure you install [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md). 
```
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```
Please see [CA Bodies repo](https://github.com/facebookresearch/ca_body) for more details on the renderer.

# Download data and models
To download any of the datasets, you can find them at `https://github.com/facebookresearch/audio2photoreal/releases/download/v1.0/<person_id>.zip`, where you can replace `<person_id>` with any of `PXB184`, `RLW104`, `TXB805`, or `GQS883`.
Download over the command line can be done with this commands.
```
curl -L https://github.com/facebookresearch/audio2photoreal/releases/download/v1.0/<person_id>.zip -o <person_id>.zip
unzip <person_id>.zip -d dataset/
rm <person_id>.zip
```
:point_down: To download *all* of the datasets, you can simply run the following which will download and unpack all the models.
```
sh scripts/download_alldatasets.sh
```

Similarly, to download any of the models, you can find them at `http://audio2photoreal_models.berkeleyvision.org/<person_id>_models.tar`.
```
# download the motion generation
wget http://audio2photoreal_models.berkeleyvision.org/<person_id>_models.tar
tar xvf <person_id>_models.tar
rm <person_id>_models.tar

# download the body decoder/rendering assets and place them in the right place
mkdir -p checkpoints/ca_body/data/
wget https://github.com/facebookresearch/ca_body/releases/download/v0.0.1-alpha/<person_id>.tar.gz
tar xvf <person_id>.tar.gz --directory checkpoints/ca_body/data/
rm <person_id>.tar.gz
```
:point_down: You can also download all of the models with this script:
```
sh scripts/download_allmodels.sh
```
The above model script will download *both* the models for motion generation and the body decoder/rendering models. Please view the script for more details.

### Dataset 
Once the dataset is downloaded and unzipped (via `scripts/download_datasets.sh`), it should unfold into the following directory structure:
```
|-- dataset/
    |-- PXB184/
        |-- data_stats.pth 
        |-- scene01_audio.wav
        |-- scene01_body_pose.npy
        |-- scene01_face_expression.npy
        |-- scene01_missing_face_frames.npy
        |-- ...
        |-- scene30_audio.wav
        |-- scene30_body_pose.npy
        |-- scene30_face_expression.npy
        |-- scene30_missing_face_frames.npy
    |-- RLW104/
    |-- TXB805/
    |-- GQS883/
```
Each of the four participants (`PXB184`, `RLW104`, `TXB805`, `GQS883`) should have independent "scenes" (1 to 26 or so).
For each scene, there are 3 types of data annotations that we save. 
```
*audio.wav: wavefile containing the raw audio (two channels, 1600*T samples) at 48kHz; channel 0 is the audio associated with the current person, channel 1 is the audio associated with their conversational partner.

*body_pose.npy: (T x 104) array of joint angles in a kinematic skeleton. Not all of the joints are represented with 3DoF. Each 104-d vector can be used to reconstruct a full-body skeleton.

*face_expression.npy: (T x 256) array of facial codes, where each 256-d vector reconstructs a face mesh.

*missing_face_frames.npy: List of indices (t) where the facial code is missing or corrupted. 

data_stats.pth: carries the mean and std for each modality of each person.
```

For the train/val/test split the indices are defined in `data_loaders/data.py` as:
```
train_idx = list(range(0, len(data_dict["data"]) - 6))
val_idx = list(range(len(data_dict["data"]) - 6, len(data_dict["data"]) - 4))
test_idx = list(range(len(data_dict["data"]) - 4, len(data_dict["data"])))
```
for any of the four dataset participants we train on.

### Visualize ground truth
If you've properly installed the rendering requirements, you can then visualize the full dataset with the following command:
```
python -m visualize.render_anno 
    --save_dir <path/to/save/dir> 
    --data_root <path/to/data/root> 
    --max_seq_length <num>
```

The videos will be chunked lengths according to specified `--max_seq_length` arg, which you can specify (the default is 600).

:point_down: For example, to visualize ground truth annotations for `PXB184`, you can run the following.
```
python -m visualize.render_anno --save_dir vis_anno_test --data_root dataset/PXB184 --max_seq_length 600
```

### Pretrained models
We train person-specific models, so each person should have an associated directory. For instance, for `PXB184`, their complete models should unzip into the following structure.
```
|-- checkpoints/
    |-- diffusion/
        |-- c1_face/
            |-- args.json
            |-- model:09d.pt
        |-- c1_pose/
            |-- args.json
            |-- model:09d.pt
    |-- guide/
        |-- c1_pose/
            |-- args.json
            |-- checkpoints/
                |-- iter-:07d.pt
    |-- vq/
        |-- c1_pose/
            |-- args.json
            |-- net_iter:06d.pth
```
There are 4 models for each person and each model has an associated `args.json`.
1. a face diffusion model that outputs 256 facial codes conditioned on audio
2. a pose diffusion model that outputs 104 joint rotations conditioned on audio and guide poses
3. a guide vq pose model that outputs vq tokens conditioned on audio at 1 fps
4. a vq encoder-decoder model that vector quantizes the continuous 104-d pose space.

# Running the pretrained models 
To run the actual models, you will need to run the pretrained models and generate the associated results files before visualizing them. 

### Face generation
To generate the results file for the face, 
```
python -m sample.generate 
    --model_path <path/to/model> 
    --num_samples <xsamples> 
    --num_repetitions <xreps> 
    --timestep_respacing ddim500 
    --guidance_param 10.0
```

The `<path/to/model>` should be the path to the diffusion model that is associated with generating the face. 
E.g. for participant `PXB184`, the path might be `./checkpoints/diffusion/c1_face/model000155000.pt`
The other parameters are:
```
--num_samples: number of samples to generate. To sample the full dataset, use 56 (except for TXB805, whcih is 58).
--num_repetitions: number of times to repeat the sampling, such that total number of sequences generated is (num_samples * num_repetitions). 
--timestep_respacing: how many diffusion steps to take. Format will always be ddim<number>.
--guidance_param: how influential the conditioning is on the results. I usually use range 2.0-10.0, and tend towards higher for the face.
```

:point_down: A full example of running the face model for `PXB184` with the provided pretrained models would then be:
```
python -m sample.generate --model_path checkpoints/diffusion/c1_face/model000155000.pt --num_samples 10 --num_repetitions 5 --timestep_respacing ddim500 --guidance_param 10.0
```
This generates 10 samples from the dataset 1 time. The output results file will be saved to:
`./checkpoints/diffusion/c1_face/samples_c1_face_000155000_seed10_/results.npy`

### Body generation
To generate the corresponding body, it will be very similar to generating the face, except now we have to feed in the model for generating the guide poses as well.
```
python -m sample.generate 
    --model_path <path/to/model> 
    --resume_trans <path/to/guide/model> 
    --num_samples <xsamples> 
    --num_repetitions <xreps> 
    --timestep_respacing ddim500 
    --guidance_param 10.0
```

:point_down: Here, `<path/to/guide/model>` should point to the guide transformer. The full command would be:
```
python -m sample.generate --model_path checkpoints/diffusion/c1_pose/model000340000.pt --resume_trans checkpoints/guide/c1_pose/checkpoints/iter-0100000.pt --num_samples 10 --num_repetitions 5 --timestep_respacing ddim500 --guidance_param 10.0
```
Similarly, the output will be saved to:
`./checkpoints/diffusion/c1_pose/samples_c1_pose_000340000_seed10_guide_iter-0100000.pt/results.npy`

### Visualization
On the body generation side of things, you can also optionally pass in the `--plot` flag in order to render out the photorealistic avatar. You will also need to pass in the corresponding generated face codes with the `--face_codes` flag.
Optionally, if you already have the poses precomputed, you an also pass in the generated body with the `--pose_codes` flag.
This will save videos in the same directory as where the body's `results.npy` is stored. 

:point_down: An example of the full command with *the three new flags added is*:
```
python -m sample.generate --model_path checkpoints/diffusion/c1_pose/model000340000.pt --resume_trans checkpoints/guide/c1_pose/checkpoints/iter-0100000.pt --num_samples 10 --num_repetitions 5 --timestep_respacing ddim500 --guidance_param 10.0 --face_codes ./checkpoints/diffusion/c1_face/samples_c1_face_000155000_seed10_/results.npy --pose_codes ./checkpoints/diffusion/c1_pose/samples_c1_pose_000340000_seed10_guide_iter-0100000.pt/results.npy --plot
```
The remaining flags can be the same as before. For the actual rendering api, please see [Codec Avatar Body](https://github.com/facebookresearch/ca_body) for installation etc.
*Important: in order to visualize the full photorealistic avatar, you will need to run the face codes first, then pass them into the body generation code.* It will not work if you try to call generate with `--plot` for the face codes.

# Training from scratch
There are four possible models you will need to train: 1) the face diffusion model, 2) the body diffusion model, 3) the body vq vae, 4) the body guide transformer.
The only dependency is that 3) is needed for 4). All other models can be trained in parallel.

### 1) Face diffusion model
To train the face model, you will need to run the following script:
```
python -m train.train_diffusion 
    --save_dir <path/to/save/dir>
    --data_root <path/to/data/root>
    --batch_size <bs>
    --dataset social  
    --data_format face 
    --layers 8 
    --heads 8 
    --timestep_respacing ''
    --max_seq_length 600
```
Importantly, a few of the flags are as follows:
```
--save_dir: path to directory where all outputs are stored
--data_root: path to the directory of where to load the data from
--dataset: name of dataset to load; right now we only support the 'social' dataset
--data_format: set to 'face' for the face, as opposed to pose
--timestep_respacing: set to '' which does the default spacing of 1k diffusion steps
--max_seq_length: the maximum number of frames for a given sequence to train on
```
:point_down: A full example for training on person `PXB184` is:
```
python -m train.train_diffusion --save_dir checkpoints/diffusion/c1_face_test --data_root ./dataset/PXB184/ --batch_size 4 --dataset social --data_format face --layers 8 --heads 8 --timestep_respacing '' --max_seq_length 600
```

### 2) Body diffusion model
Training the body model is similar to the face model, but with the following additional parameters
```
python -m train.train_diffusion 
    --save_dir <path/to/save/dir> 
    --data_root <path/to/data/root>
    --lambda_vel <num>
    --batch_size <bs> 
    --dataset social 
    --add_frame_cond 1 
    --data_format pose 
    --layers 6 
    --heads 8 
    --timestep_respacing '' 
    --max_seq_length 600
```
The flags that differ from the face training are as follows:
```
--lambda_vel: additional auxilary loss for training with velocity
--add_frame_cond: set to '1' for 1 fps. if not specified, it will default to 30 fps.
--data_format: set to 'pose' for the body, as opposed to face
```
:point_down: A full example for training on person `PXB184` is:
```
python -m train.train_diffusion --save_dir checkpoints/diffusion/c1_pose_test --data_root ./dataset/PXB184/ --lambda_vel 2.0 --batch_size 4 --dataset social --add_frame_cond 1 --data_format pose --layers 6 --heads 8 --timestep_respacing '' --max_seq_length 600
```

### 3) Body VQ VAE
To train a vq encoder-decoder, you will need to run the following script:
```
python -m train.train_vq 
    --out_dir <path/to/out/dir> 
    --data_root <path/to/data/root>
    --batch_size <bs>
    --lr 1e-3 
    --code_dim 1024 
    --output_emb_width 64 
    --depth 4 
    --dataname social 
    --loss_vel 0.0 
    --add_frame_cond 1 
    --data_format pose 
    --max_seq_length 600
```
:point_down: For person `PXB184`, it would be:
```
python -m train.train_vq --out_dir checkpoints/vq/c1_vq_test --data_root ./dataset/PXB184/ --lr 1e-3 --code_dim 1024 --output_emb_width 64 --depth 4 --dataname social --loss_vel 0.0 --data_format pose --batch_size 4 --add_frame_cond 1 --max_seq_length 600
```

### 4) Body guide transformer
Once you have the vq trained from 3) you can then pass it in to train the body guide pose transformer:
```
python -m train.train_guide 
    --out_dir <path/to/out/dir>
    --data_root <path/to/data/root>
    --batch_size <bs>
    --resume_pth <path/to/vq/model>
    --add_frame_cond 1 
    --layers 6 
    --lr 2e-4 
    --gn 
    --dim 64 
```
:point_down: For person `PXB184`, it would be:
```
python -m train.train_guide --out_dir checkpoints/guide/c1_trans_test --data_root ./dataset/PXB184/ --batch_size 4 --resume_pth checkpoints/vq/c1_vq_test/net_iter300000.pth --add_frame_cond 1 --layers 6 --lr 2e-4 --gn --dim 64
```

After training these 4 models, you can now follow the ["Running the pretrained models"](#running-the-pretrained-models) section to generate samples and visualize results.

You can also visualize the corresponding ground truth sequences by passing in the `--render_gt` flag.


# License
The code and dataset are released under [CC-NC 4.0 International license](https://github.com/facebookresearch/audio2photoreal/blob/main/LICENSE).
