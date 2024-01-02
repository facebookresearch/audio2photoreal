"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import fairseq
import torch as th
import torchaudio as ta

wav2vec_model_path = "./assets/wav2vec_large.pt"


def weights_init(m):
    if isinstance(m, th.nn.Conv1d):
        th.nn.init.xavier_uniform_(m.weight)
        try:
            th.nn.init.constant_(m.bias, 0.01)
        except:
            pass


class Wav2VecEncoder(th.nn.Module):
    def __init__(self):
        super().__init__()
        self.resampler = ta.transforms.Resample(orig_freq=48000, new_freq=16000)
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [wav2vec_model_path]
        )
        self.wav2vec_model = model[0]

    def forward(self, audio: th.Tensor):
        """
        :param audio: B x T x 1600
        :return: B x T_wav2vec x 512
        """
        audio = audio.view(audio.shape[0], audio.shape[1] * 1600)
        audio = self.resampler(audio)
        audio = th.cat(
            [th.zeros(audio.shape[0], 320, device=audio.device), audio], dim=-1
        )  # zero padding on the left
        x = self.wav2vec_model.feature_extractor(audio)
        x = self.wav2vec_model.feature_aggregator(x)
        x = x.permute(0, 2, 1).contiguous()
        return x


class Wav2VecDownsampler(th.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = th.nn.Conv1d(512, 512, kernel_size=3)
        self.conv2 = th.nn.Conv1d(512, 512, kernel_size=3)
        self.norm = th.nn.LayerNorm(512)

    def forward(self, x: th.Tensor, target_length: int):
        """
        :param x: B x T x 512 tensor containing wav2vec features at 100Hz
        :return: B x target_length x 512 tensor containing downsampled wav2vec features at 30Hz
        """
        x = x.permute(0, 2, 1).contiguous()
        # first conv
        x = th.nn.functional.pad(x, pad=(2, 0))
        x = th.nn.functional.relu(self.conv1(x))
        # first downsampling
        x = th.nn.functional.interpolate(x, size=(x.shape[-1] + target_length) // 2)
        # second conv
        x = th.nn.functional.pad(x, pad=(2, 0))
        x = self.conv2(x)
        # second downsampling
        x = th.nn.functional.interpolate(x, size=target_length)
        # layer norm
        x = x.permute(0, 2, 1).contiguous()
        x = self.norm(x)
        return x


class AudioTcn(th.nn.Module):
    def __init__(
        self,
        encoding_dim: int = 128,
        use_melspec: bool = True,
        use_wav2vec: bool = True,
    ):
        """
        :param encoding_dim: size of encoding
        :param use_melspec: extract mel spectrogram features as input
        :param use_wav2vec: extract wav2vec features as input
        """
        super().__init__()
        self.encoding_dim = encoding_dim
        self.use_melspec = use_melspec
        self.use_wav2vec = use_wav2vec

        if use_melspec:
            # hop_length=400 -> two feature vectors per visual frame (downsampling to 24kHz -> 800 samples per frame)
            self.melspec = th.nn.Sequential(
                ta.transforms.Resample(orig_freq=48000, new_freq=24000),
                ta.transforms.MelSpectrogram(
                    sample_rate=24000,
                    n_fft=1024,
                    win_length=800,
                    hop_length=400,
                    n_mels=80,
                ),
            )

        if use_wav2vec:
            model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                [wav2vec_model_path]
            )
            self.wav2vec_model = model[0]
            self.wav2vec_model.eval()
            self.wav2vec_postprocess = th.nn.Conv1d(512, 256, kernel_size=3)
            self.wav2vec_postprocess.apply(lambda x: weights_init(x))

        # temporal model
        input_dim = 0 + (160 if use_melspec else 0) + (256 if use_wav2vec else 0)
        self.layers = th.nn.ModuleList(
            [
                th.nn.Conv1d(
                    input_dim, max(256, encoding_dim), kernel_size=3, dilation=1
                ),  # 2 (+1)
                th.nn.Conv1d(
                    max(256, encoding_dim), encoding_dim, kernel_size=3, dilation=2
                ),  # 4 (+1)
                th.nn.Conv1d(
                    encoding_dim, encoding_dim, kernel_size=3, dilation=3
                ),  # 6 (+1)
                th.nn.Conv1d(
                    encoding_dim, encoding_dim, kernel_size=3, dilation=1
                ),  # 2 (+1)
                th.nn.Conv1d(
                    encoding_dim, encoding_dim, kernel_size=3, dilation=2
                ),  # 4 (+1)
                th.nn.Conv1d(
                    encoding_dim, encoding_dim, kernel_size=3, dilation=3
                ),  # 6 (+1)
            ]
        )
        self.layers.apply(lambda x: weights_init(x))
        self.receptive_field = 25

        self.final = th.nn.Conv1d(encoding_dim, encoding_dim, kernel_size=1)
        self.final.apply(lambda x: weights_init(x))

    def forward(self, audio):
        """
        :param audio: B x T x 1600 tensor containing audio samples for each frame
        :return: B x T x encoding_dim tensor containing audio encodings for each frame
        """
        B, T = audio.shape[0], audio.shape[1]

        # preprocess raw audio signal to extract feature vectors
        audio = audio.view(B, T * 1600)
        x_mel, x_w2v = th.zeros(B, 0, T).to(audio.device), th.zeros(B, 0, T).to(
            audio.device
        )
        if self.use_melspec:
            x_mel = self.melspec(audio)[:, :, 1:].contiguous()
            x_mel = th.log(x_mel.clamp(min=1e-10, max=None))
            x_mel = (
                x_mel.permute(0, 2, 1)
                .contiguous()
                .view(x_mel.shape[0], T, 160)
                .permute(0, 2, 1)
                .contiguous()
            )
        if self.use_wav2vec:
            with th.no_grad():
                x_w2v = ta.functional.resample(audio, 48000, 16000)
                x_w2v = self.wav2vec_model.feature_extractor(x_w2v)
                x_w2v = self.wav2vec_model.feature_aggregator(x_w2v)
            x_w2v = self.wav2vec_postprocess(th.nn.functional.pad(x_w2v, pad=[2, 0]))
            x_w2v = th.nn.functional.interpolate(
                x_w2v, size=T, align_corners=True, mode="linear"
            )
        x = th.cat([x_mel, x_w2v], dim=1)

        # process signal with TCN
        x = th.nn.functional.pad(x, pad=[self.receptive_field - 1, 0])
        for layer_idx, layer in enumerate(self.layers):
            y = th.nn.functional.leaky_relu(layer(x), negative_slope=0.2)
            if self.training:
                y = th.nn.functional.dropout(y, 0.2)
            if x.shape[1] == y.shape[1]:
                x = (x[:, :, -y.shape[-1] :] + y) / 2.0  # skip connection
            else:
                x = y

        x = self.final(x)
        x = x.permute(0, 2, 1).contiguous()

        return x
