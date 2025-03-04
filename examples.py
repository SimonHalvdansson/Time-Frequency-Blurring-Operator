#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F


# Audio and spectrogram parameters
TARGET_LENGTH = 16000  # 1 second @ 16kHz
N_FFT = 256
HOP_LENGTH = 128

# Global Kernel parameters
KERNEL_SIZE = 41
GAUSS_SIGMA_X = 1.5
GAUSS_SIGMA_Y = 1.5

# Parameters for the translated Gaussian kernel (more aggressive shift)
TRANSLATED_SIGMA = 1.0
TRANSLATED_SHIFT_X = 16
TRANSLATED_SHIFT_Y = -16

CIRCLE_RADIUS = KERNEL_SIZE // 2 // 5

# ------------------------------
# Utility Functions
# ------------------------------

def pad_waveform(waveform, target_length):
    """Pads waveform with zeros if it is shorter than target_length."""
    current_length = waveform.shape[-1]
    if current_length < target_length:
        padding_length = target_length - current_length
        waveform = F.pad(waveform, (0, padding_length))
    return waveform

def waveform_to_spectrogram(waveform, n_fft=N_FFT, win_length=None, hop_length=HOP_LENGTH):
    """
    Converts a waveform to a spectrogram (in dB) and normalizes it.
    The output tensor has shape (1, freq_bins, time_steps).
    """
    spec_trans = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=1.0,
    )
    waveform = waveform.reshape(1, -1)
    spec = spec_trans(waveform)
    spec = torchaudio.transforms.AmplitudeToDB()(spec)
    spec -= spec.min()
    spec /= spec.max()
    return spec

def get_random_waveform(data_path, target_length=TARGET_LENGTH):
    """
    Selects a random .wav file from a random subdirectory in data_path,
    loads it, and crops/pads it to target_length samples.
    """
    subdirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    chosen_subdir = random.choice(subdirs)
    wav_files = list(Path(os.path.join(data_path, chosen_subdir)).glob('*.wav'))
    if not wav_files:
        raise ValueError(f"No .wav files found in {os.path.join(data_path, chosen_subdir)}")
    chosen_file = random.choice(wav_files)
    waveform, sr = torchaudio.load(str(chosen_file))
    # Crop or pad to get exactly target_length samples
    if waveform.shape[1] > target_length:
        start = random.randint(0, waveform.shape[1] - target_length)
        waveform = waveform[:, start:start+target_length]
    elif waveform.shape[1] < target_length:
        waveform = pad_waveform(waveform, target_length)
    return waveform, sr

# ------------------------------
# Custom Kernel Generators
# ------------------------------

def square_kernel(size=KERNEL_SIZE, inner_size=4):
    """
    Creates a square kernel of shape (size, size) with a centered square of ones and zeros elsewhere.
    The width of the central square is defined by inner_size (default: size//2).
    The kernel is normalized to sum to 1.
    """
    if inner_size is None:
        inner_size = size // 2
    kernel = torch.zeros((size, size))
    start = (size - inner_size) // 2
    kernel[start:start+inner_size, start:start+inner_size] = 1
    kernel = kernel / kernel.sum()
    return kernel

def horizontal_line_kernel(size=KERNEL_SIZE, height=3):
    """
    Creates a horizontal line kernel of shape (size, size) with a horizontal band of ones 
    (of given height) centered vertically, and zeros elsewhere. The kernel is normalized to sum to 1.
    """
    kernel = torch.zeros((size, size))
    start = (size - height) // 2
    kernel[start:start+height, :] = 1
    kernel = kernel / kernel.sum()
    return kernel

def vertical_line_kernel(size=KERNEL_SIZE, height=3):
    """
    Creates a vertical line kernel of shape (size, size) with a vertical band of ones 
    (of given height) centered horizontally, and zeros elsewhere. The kernel is normalized to sum to 1.
    """
    kernel = torch.zeros((size, size))
    start = (size - height) // 2
    kernel[:, start:start+height] = 1
    kernel = kernel / kernel.sum()
    return kernel

def translated_gaussian_kernel(size=KERNEL_SIZE, sigma=TRANSLATED_SIGMA,
                               shift_x=TRANSLATED_SHIFT_X, shift_y=TRANSLATED_SHIFT_Y):
    """
    Creates a 2D Gaussian kernel that is translated (shifted) by shift_x and shift_y.
    The kernel is normalized so that its sum is 1.
    """
    ax = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0 - shift_x
    ay = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0 - shift_y
    xx, yy = torch.meshgrid(ax, ay, indexing='xy')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel

def circle_kernel(size=KERNEL_SIZE, radius=CIRCLE_RADIUS):
    """
    Creates a circular kernel (ones inside a circle, zeros outside)
    normalized so that its sum is 1.
    """
    center = (size - 1) / 2.0
    yy, xx = torch.meshgrid(torch.arange(size, dtype=torch.float32), 
                              torch.arange(size, dtype=torch.float32), indexing='ij')
    distance = torch.sqrt((xx - center)**2 + (yy - center)**2)
    kernel = (distance <= radius).float()
    kernel = kernel / kernel.sum()
    return kernel

# ------------------------------
# Blurring Functions
# ------------------------------

def apply_blur_with_kernel(tensor, kernel):
    """
    Applies a 2D convolution to 'tensor' using the provided 'kernel'.
    The input 'tensor' should be of shape (1, H, W) for a spectrogram or an STFT component.
    """    
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    kernel_expanded = kernel.unsqueeze(0).unsqueeze(0)  # shape: (1,1,kh,kw)
    kernel_expanded = kernel_expanded.expand(tensor.shape[0], -1, -1, -1)
    # Pad using zero padding (constant value 0) to reduce edge artifacts.
    padded_tensor = F.pad(tensor, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
    blurred = F.conv2d(padded_tensor, kernel_expanded, stride=1, groups=tensor.shape[0])
    return blurred

def gaussian_kernel(kernel_size=KERNEL_SIZE, sigma_x=GAUSS_SIGMA_X, sigma_y=GAUSS_SIGMA_Y):
    """
    Creates a 2D Gaussian kernel using separable 1D Gaussians.
    The kernel is normalized to sum to 1.
    """
    x = torch.linspace(-kernel_size // 2 + 1, kernel_size // 2, kernel_size)
    gauss_kernel_x = torch.exp(-x**2 / (2 * sigma_x**2))
    gauss_kernel_y = torch.exp(-x**2 / (2 * sigma_y**2))
    gauss_kernel_x /= gauss_kernel_x.sum()
    gauss_kernel_y /= gauss_kernel_y.sum()
    kernel = gauss_kernel_x.view(1, kernel_size) * gauss_kernel_y.view(kernel_size, 1)
    kernel = kernel / kernel.sum()
    return kernel

def blur_tensor(tensor, kernel_size=KERNEL_SIZE, sigma_x=GAUSS_SIGMA_X, sigma_y=GAUSS_SIGMA_Y):
    """
    Blurs the input tensor using a Gaussian kernel defined by kernel_size and sigma.
    """
    kernel = gaussian_kernel(kernel_size, sigma_x, sigma_y)
    return apply_blur_with_kernel(tensor, kernel)

def add_spec_blur(spec, kernel):
    """
    Applies spectrogram blurring to 'spec' using the provided 'kernel'
    and then normalizes the result.
    """
    blurred = apply_blur_with_kernel(spec, kernel)
    blurred -= blurred.min()
    blurred /= blurred.max()
    return blurred

def blur_waveform_complex(waveform, kernel, window):
    """
    Computes the complex STFT of 'waveform', applies the provided 'kernel'
    to the real and imaginary parts separately, and reconstructs the waveform.
    """
    stft_complex = torch.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window, return_complex=True)
    stft_real_blurred = apply_blur_with_kernel(stft_complex.real, kernel)
    stft_imag_blurred = apply_blur_with_kernel(stft_complex.imag, kernel)
    stft_blurred = stft_real_blurred + 1j * stft_imag_blurred
    waveform_recon = torch.istft(stft_blurred, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window, length=TARGET_LENGTH)
    return waveform_recon

# ------------------------------
# Plotting Functions
# ------------------------------

def plot_operator_figures_stft():
    """
    Computes a complex-valued STFT of a waveform, blurs it with various kernels,
    reconstructs the waveform using inverse STFT, and then plots a 3x5 figure.
    
    Columns correspond to the five different kernels. The rows represent:
      - Top row: STFT-based blurred spectrogram.
      - Middle row: Spectrogram (magnitude)-based blurred spectrogram.
      - Bottom row: Kernel visualization.
    """
    # Load one random 1-sec waveform from SpeechCommands.
    data_path = os.path.join('data', 'SpeechCommands', 'speech_commands_v0.02')
    waveform, sr = get_random_waveform(data_path, TARGET_LENGTH)
    
    # Create a Hann window for STFT and inverse STFT.
    window = torch.hann_window(N_FFT)
    
    # Compute the complex-valued STFT (with return_complex=True).
    stft_complex = torch.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window, return_complex=True)
    
    # Compute the raw spectrogram (for spectrogram-based blurring)
    raw_spec = waveform_to_spectrogram(waveform)
    
    # Define the kernels.
    kernels = {
        'Square': square_kernel(),
        'Horizontal': horizontal_line_kernel(),
        'Vertical': vertical_line_kernel(),
        'Translated Gauss': translated_gaussian_kernel(),
        'Circle': circle_kernel()
    }
    
    # Dictionaries to store the results.
    stft_blur_specs = {}
    spec_blur_specs = {}
    
    for name, kernel in kernels.items():
        # --- STFT-based blurring ---
        stft_real_blurred = apply_blur_with_kernel(stft_complex.real, kernel)
        stft_imag_blurred = apply_blur_with_kernel(stft_complex.imag, kernel)
        stft_blurred = stft_real_blurred + 1j * stft_imag_blurred
        
        # Reconstruct the waveform using inverse STFT.
        waveform_recon = torch.istft(stft_blurred, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window, length=TARGET_LENGTH)
        spec_recon_stft = waveform_to_spectrogram(waveform_recon)
        stft_blur_specs[name] = spec_recon_stft.squeeze().numpy()
        
        # --- Spectrogram-based blurring ---
        spec_recon_spec = add_spec_blur(raw_spec, kernel)
        spec_blur_specs[name] = spec_recon_spec.squeeze().numpy()
    
    # Create a 3 (rows) x 5 (columns) subplot grid.
    fig, axes = plt.subplots(nrows=3, ncols=len(kernels), figsize=(15, 12), dpi=500)
    fig.suptitle('Blurring Comparisons: STFT vs. Spectrogram Convolution', fontsize=16)
    
    # Iterate over kernels to populate columns.
    for j, (name, kernel) in enumerate(kernels.items()):
        # Top row: STFT-based blurred spectrogram.
        ax_top = axes[0, j]
        im_top = ax_top.imshow(stft_blur_specs[name], origin='lower', aspect='auto')
        ax_top.set_title(name)
        ax_top.set_ylabel('STFT Blurred')
        ax_top.set_xlabel('Time')
        
        # Middle row: Spectrogram-based blurred spectrogram.
        ax_mid = axes[1, j]
        im_mid = ax_mid.imshow(spec_blur_specs[name], origin='lower', aspect='auto')
        ax_mid.set_ylabel('Spec Blurred')
        ax_mid.set_xlabel('Time')
        
        # Bottom row: Kernel visualization.
        ax_bot = axes[2, j]
        im_bot = ax_bot.imshow(kernel.numpy(), cmap='viridis', origin='lower', aspect='auto', interpolation='nearest')
        ax_bot.set_ylabel('Kernel')
        ax_bot.set_xlabel('Kernel X')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_white_noise_comparison():
    """
    Creates a 1x3 figure comparing:
      - Left: Raw spectrogram of white noise (no blurring).
      - Middle: Spectrogram blurred using complex STFT processing with a Gaussian kernel.
      - Right: Spectrogram blurred using the magnitude-based add_spec_blur (using the same kernel).
    """
    # Generate white noise (1 sec)
    white_noise = torch.randn(1, TARGET_LENGTH)
    
    # Create a Hann window for STFT processing.
    window = torch.hann_window(N_FFT)
    
    # Compute raw spectrogram (no blurring)
    raw_spec = waveform_to_spectrogram(white_noise)
    
    # Compute a single Gaussian kernel to be shared.
    kernel_gauss = gaussian_kernel()
    
    # Middle: blur using complex STFT processing (apply the kernel to real and imaginary parts)
    waveform_blur_complex = blur_waveform_complex(white_noise, kernel_gauss, window)
    noise_blurred_complex = waveform_to_spectrogram(waveform_blur_complex)
    
    # Right: blur using the magnitude-based method with add_spec_blur (using same kernel)
    spec_noise = waveform_to_spectrogram(white_noise)
    noise_blurred_spec = add_spec_blur(spec_noise, kernel_gauss)
    
    # Create 1x3 plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=500)
    
    # Left: Raw spectrogram
    axes[0].imshow(raw_spec.squeeze().numpy(), origin='lower', aspect='auto')
    axes[0].set_title('Raw Spectrogram')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Frequency')
    
    # Middle: Gaussian Window Blur (complex STFT-based)
    axes[1].imshow(noise_blurred_complex.squeeze().numpy(), origin='lower', aspect='auto')
    axes[1].set_title('Gaussian Window Blur\n(Complex STFT)')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Frequency')
    
    # Right: Magnitude-based blur via add_spec_blur (using same kernel)
    axes[2].imshow(noise_blurred_spec.squeeze().numpy(), origin='lower', aspect='auto')
    axes[2].set_title('Spectrogram Blurring\n(Magnitude-based)')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Frequency')
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

# ------------------------------
# Main Execution
# ------------------------------

if __name__ == '__main__':
    # Plot the STFT-based operator figures with the new 3x5 layout.
    plot_operator_figures_stft()
    
    # Plot the white noise comparison figure.
    plot_white_noise_comparison()
