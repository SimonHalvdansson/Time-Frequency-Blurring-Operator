#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 10:09:33 2024

@author: simohal
"""

import torch
import torchaudio

import matplotlib.pyplot as plt

from main import waveform_to_spectrogram
from main import add_stft_conv
from main import plot_spectrogram


def loc_op(wf):
    n_fft = 2048
    win_length = None
    hop_length = 128
    
    stft_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=None
    )
    
    # Apply the transform to the audio signal
    wf = wf.reshape(1, -1)  # Make sure the input tensor has the correct shape
    
    stft = stft_transform(wf)
    
    square_width = int(0.45 * stft.shape[1])  # 75% of the width
    square_height = int(0.35 * stft.shape[2])  # 65% of the height
    
    start_width = 0
    start_height = (stft.shape[2] - square_height) // 2
    
    mask_tensor = torch.zeros(stft.shape)
    
    mask_tensor[:, start_width:start_width+square_width, start_height:start_height+square_height] = 1
    
    stft = stft * mask_tensor

    istft_transform = torchaudio.transforms.InverseSpectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect"
    )
    
    recon_wf = istft_transform(stft)
    
    return recon_wf



# Load the WAV file as a tensor
file_path = 'data/SpeechCommands/speech_commands_v0.02/forward/15dd287d_nohash_4.wav'
wf, sr = torchaudio.load(file_path)

wf_blur = add_stft_conv(wf, sigma_time=2.2, sigma_freq=6)
wf_loc = loc_op(wf)

spec = waveform_to_spectrogram(wf)
spec_blur = waveform_to_spectrogram(wf_blur)
spec_loc = waveform_to_spectrogram(wf_loc)



plot_spectrogram(spec_blur / (spec + 0.8), title="Blur quotient")



fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True, dpi=300)

axes[0].imshow(spec.squeeze(), aspect="auto", origin="lower", extent=[0, 1, 0, 256])
axes[1].imshow(spec_loc.squeeze(), aspect="auto", origin="lower", extent=[0, 1, 0, 256])

axes[0].set_title("Original spectrogram")
axes[1].set_title("Localized spectrogram")

axes[0].set_ylabel('Frequency')
axes[0].set_xlabel('Time')
axes[1].set_xlabel('Time')


fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True, dpi=300)

axes[0].imshow(spec.squeeze(), aspect="auto", origin="lower", extent=[0, 1, 0, 256])
axes[1].imshow(spec_blur.squeeze(), aspect="auto", origin="lower", extent=[0, 1, 0, 256])

axes[0].set_title("Original spectrogram")
axes[1].set_title("Blurred spectrogram")

axes[0].set_ylabel('Frequency')
axes[0].set_xlabel('Time')
axes[1].set_xlabel('Time')





