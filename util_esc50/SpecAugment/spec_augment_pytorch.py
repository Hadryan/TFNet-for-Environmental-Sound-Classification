# # Reference : https://github.com/DemisEom/SpecAugment
# Copyright 2019 RnD at Spoon Radio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""SpecAugment Implementation for Tensorflow.
Related paper : https://arxiv.org/pdf/1904.08779.pdf

In this paper, show summarized parameters by each open datasets in Tabel 1.
-----------------------------------------
Policy | W  | F  | m_F |  T  |  p  | m_T
-----------------------------------------
None   |  0 |  0 |  -  |  0  |  -  |  -
-----------------------------------------
LB     | 80 | 27 |  1  | 100 | 1.0 | 1
-----------------------------------------
LD     | 80 | 27 |  2  | 100 | 1.0 | 2
-----------------------------------------
SM     | 40 | 15 |  2  |  70 | 0.2 | 2
-----------------------------------------
SS     | 40 | 27 |  2  |  70 | 0.2 | 2
-----------------------------------------
LB : LibriSpeech basic
LD : LibriSpeech double
SM : Switchboard mild
SS : Switchboard strong
"""

import librosa
import librosa.display
import math
import numpy as np
import random
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def spec_augment(mel_spectrogram, using_time_warping=False, using_frequency_masking=False, using_time_masking=False,
                 frequency_masking_para=4, time_masking_para=4,
                 frequency_mask_num=2, time_mask_num=2):
    """Spec augmentation Calculation Function.

    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.

    # Arguments:
      mel_spectrogram(numpy array): audio file path of you want to warping and masking.
      time_warping_para(float): Augmentation parameter, "time warp parameter W".
        If none, default = 80 for LibriSpeech.
      frequency_masking_para(float): Augmentation parameter, "frequency mask parameter F"
        If none, default = 100 for LibriSpeech.
      time_masking_para(float): Augmentation parameter, "time mask parameter T"
        If none, default = 27 for LibriSpeech.
      frequency_mask_num(float): number of frequency masking lines, "m_F".
        If none, default = 1 for LibriSpeech.
      time_mask_num(float): number of time masking lines, "m_T".
        If none, default = 1 for LibriSpeech.

    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """
    # mel_spectrogram:
    # (batch_size, times_steps, freq_bins)
    # v : freq_bins
    v = mel_spectrogram.shape[3]
    # tau : times_steps
    tau = mel_spectrogram.shape[2]
    num = mel_spectrogram.shape[0]


    warped_mel_spectrogram = mel_spectrogram
    # Step 1 : Time warping (TO DO...)
    if using_time_warping:
        for n in range(num):
            for i in range(tau):
                for j in range(v):
                    offset_x = random.randint(0, i-1)
                    warped_mel_spectrogram[n, :, i, j] = mel_spectrogram[n, :, (i + offset_x) % tau, j]


    # Step 2 : Frequency masking
    if using_frequency_masking:
        for n in range(num):
            for i in range(frequency_mask_num):
                f = np.random.uniform(low=0.0, high=frequency_masking_para)
                f = int(f)
                f0 = random.randint(0, v - f)
                warped_mel_spectrogram[n, :, :, f0:f0 + f] = 0

    # Step 3 : Time masking
    if using_time_masking:
        for n in range(num):
            for i in range(time_mask_num):
                t = np.random.uniform(low=0.0, high=time_masking_para)
                t = int(t)
                t0 = random.randint(0, tau - t)
                warped_mel_spectrogram[n, :, t0:t0 + t, :] = 0

    return warped_mel_spectrogram


def visualization_spectrogram(mel_spectrogram, title):
    """visualizing result of SpecAugment

    # Arguments:
      mel_spectrogram(ndarray): mel_spectrogram to visualize.
      title(String): plot figure's title
    """
    # Show mel-spectrogram using librosa's specshow.
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), y_axis='mel', fmax=14000, x_axis='time')
    # plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

