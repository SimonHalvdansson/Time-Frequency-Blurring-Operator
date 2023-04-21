# -*- coding: utf-8 -*-

import os
import torchaudio
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchinfo import summary
from functools import partial
import random
from scipy.ndimage import gaussian_filter


def download_and_save_data():
    default_dir = os.getcwd()
    folder = 'data'
    print(f'Data directory will be: {default_dir}/{folder}')
    
    if os.path.isdir(folder):
        print("Data folder exists.")
    else:
        print("Creating folder.")
        os.mkdir(folder)
    
    torchaudio.datasets.SPEECHCOMMANDS(f'./{folder}/', download=True)
        

def load_audio_files(path: str, train_im, val_im, test_im):
    #For future, smallest class has 1557 examples
    train, val, test = [], [], []
    
    all_contents = os.listdir(path)
    subdirs = [content for content in all_contents if os.path.isdir(os.path.join(path, content))]
    
    for subdir in subdirs:
        subpath = os.path.join(path, subdir)
        
        walker = [str(p) for p in Path(subpath).glob(f'*.wav')]
        random.shuffle(walker)
                
        for i, file_path in enumerate(walker):
            waveform, sample_rate = torchaudio.load(file_path)
            
            if i < train_im:
                train.append([waveform, subdir])
            elif i < train_im + val_im:
                val.append([waveform, subdir])
            elif i < train_im + val_im + test_im:
                test.append([waveform, subdir])
            else:
                break
            
            
    return train, val, test, subdirs

def waveform_to_spectrogram(waveform):
    # Parameters for the spectrogram
    n_fft = 4096
    win_length = None
    hop_length = 256
    n_mels = 256
    
    
    # Create the MelSpectrogram transform
    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm='slaney',
        n_mels=n_mels,
    )
    
    # Apply the transform to the audio signal
    waveform = waveform.reshape(1, -1)  # Make sure the input tensor has the correct shape
    mel_spectrogram = mel_spectrogram_transform(waveform)
    
    # Convert the power spectrogram to a log scale
    log_mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
    
    # Shift and normalize the log-mel spectrogram to have values in the range [0, 1]
    log_mel_spectrogram -= log_mel_spectrogram.min()
    log_mel_spectrogram /= log_mel_spectrogram.max()

    return log_mel_spectrogram

def plot_spectrogram(spec, title=None, ylabel='Frequency', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('Time')
    im = axs.imshow(spec.squeeze(), origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show()

def plot_augmentations():
    #Plots the summary of augmentation methods presented in the paper
    train_dataset, validation_dataset, test_dataset, labels = load_audio_files('./data/SpeechCommands/speech_commands_v0.02', 1, 0, 0)
    wf = train_dataset[0][0]
    wf = pad_waveform(wf, 16000)
    
    wf_noise = add_noise(wf)
    wf_stft_conv = add_stft_conv(wf)
    
    spec = waveform_to_spectrogram(wf)
    spec_noise = waveform_to_spectrogram(wf_noise)
    spec_stft_conv = waveform_to_spectrogram(wf_stft_conv)
    spec_aug = add_spec_aug(spec)
    spec_blur = add_spec_blur(spec)
    
    spectrograms = [spec.numpy(), spec_noise.numpy(), spec_stft_conv.numpy(), spec_aug.numpy(), spec_blur.numpy()]
    titles = ['Original', 'White noise', 'STFT convolved', 'SpecAugment', 'SpecBlur']
    
    # Set up the figure and the subplots
    fig, axes = plt.subplots(1, 5, figsize=(15, 3), sharey=True, dpi=1000)
    fig.subplots_adjust(right=0.83)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.015, 0.6])
    
    # Plot the spectrograms
    for i, spec in enumerate(spectrograms):
        im = axes[i].imshow(spec.squeeze(), aspect="auto", origin="lower", extent=[0, 1, 0, 256])
        axes[i].set_title(titles[i])
        if i == 2:
            axes[i].set_xlabel("Time [s]")
        if i == 0:
            axes[i].set_ylabel("Mel frequency bin")
    
    # Add a smaller colorbar
    fig.colorbar(im, cax=cbar_ax)
    
    # Show the plot
    plt.show()
    
    
    #plot_spectrogram(spec, title = 'Original')
    #plot_spectrogram(spec_noise, title = 'Noise')
    #plot_spectrogram(spec_stft_conv, title = 'STFT convolve')
    #plot_spectrogram(spec_aug, title = 'SpecAugment')
    #plot_spectrogram(spec_blur, title = 'SpecBlur')
    
    
def pad_waveform(waveform, target_length):
    current_length = waveform.shape[-1]
    if current_length < target_length:
        padding_length = target_length - current_length
        waveform = torch.nn.functional.pad(waveform, (0, padding_length))
    return waveform

def one_hot_encode(class_label, labels):
    num_classes = len(labels)
    one_hot_vector = torch.zeros(num_classes)
    index = labels.index(class_label)
    one_hot_vector[index] = 1
    return one_hot_vector

def add_noise(waveform, energy = 0.15):
    return waveform + torch.randn(waveform.size()) * energy * waveform.max()
    
def add_spec_aug(spectrogram, F=27, T=10, num_freq_masks=2, num_time_masks=2):
    augmented_spectrogram = spectrogram.clone()
    _, num_freq_bins, num_time_bins = spectrogram.size()

    # Apply frequency masks
    for _ in range(num_freq_masks):
        f = torch.randint(0, F, (1,)).item()
        f0 = torch.randint(0, num_freq_bins - f, (1,)).item()
        augmented_spectrogram[:, f0:f0+f, :] = 0

    # Apply time masks
    for _ in range(num_time_masks):
        t = torch.randint(0, T, (1,)).item()
        t0 = torch.randint(0, num_time_bins - t, (1,)).item()
        augmented_spectrogram[:, :, t0:t0+t] = 0

    return augmented_spectrogram

def gaussian_kernel(kernel_size, sigma_x, sigma_y):
    # Create 1D Gaussian kernels for x and y dimensions
    x = torch.linspace(-kernel_size // 2 + 1, kernel_size // 2, kernel_size)
    gauss_kernel_x = torch.exp(-x**2 / (2 * sigma_x**2))
    gauss_kernel_y = torch.exp(-x**2 / (2 * sigma_y**2))

    gauss_kernel_x /= gauss_kernel_x.sum()  # Normalize the kernel for x
    gauss_kernel_y /= gauss_kernel_y.sum()  # Normalize the kernel for y

    # Create 2D Gaussian kernel
    gauss_kernel = gauss_kernel_x.view(1, 1, -1) * gauss_kernel_y.view(1, -1, 1)

    return gauss_kernel

def blur_tensor(tensor, kernel_size=11, sigma_x=1.5, sigma_y=1.5):
    assert len(tensor.shape) == 3, "Input tensor must have 3 dimensions (1, H, W)"

    kernel = gaussian_kernel(kernel_size, sigma_x, sigma_y)
    kernel = kernel.expand(tensor.shape[0], -1, -1, -1)

    # Pad the tensor to avoid boundary issues during convolution
    pad_size = kernel_size // 2
    padded_tensor = F.pad(tensor, (pad_size, pad_size, pad_size, pad_size), mode='reflect')

    # Convolve the tensor with the Gaussian kernel
    blurred_tensor = F.conv2d(padded_tensor, kernel, stride=1, padding=0, groups=tensor.shape[0])

    return blurred_tensor

def add_spec_blur(spec, sigma_time = 3.0, sigma_freq = 2.0):
    blurred = blur_tensor(spec, sigma_x = sigma_time, sigma_y = sigma_freq)
    
    blurred -= blurred.min()
    blurred /= blurred.max()
    return blurred

def add_stft_conv(wf, sigma_time = 0.8, sigma_freq = 4):
    n_fft = 4096
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
    
    blurred_stft_real = blur_tensor(stft.real, sigma_x = sigma_time, sigma_y = sigma_freq)
    blurred_stft_imag = blur_tensor(stft.imag, sigma_x = sigma_time, sigma_y = sigma_freq)
    blurred_stft = torch.complex(blurred_stft_real, blurred_stft_imag)
    
    istft_transform = torchaudio.transforms.InverseSpectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect"
    )
    
    recon_wf = istft_transform(blurred_stft)
    
    return recon_wf
    

def augment_waveforms(ds, noise = False, stft_conv = False):
    aug_ds = []
    
    for X, Y in ds:
        if noise:  
            aug_ds.append([add_noise(X), Y])
        if stft_conv:
            aug_ds.append([add_stft_conv(X), Y])
        
    return aug_ds

def augment_spectrogram(ds, spec_aug = False, spec_blur = False):
    aug_ds = []
    
    for X, Y in ds:
        if spec_aug:  
            aug_ds.append([add_spec_aug(X), Y])

        if spec_blur:
            aug_ds.append([add_spec_blur(X), Y])
                    
    return aug_ds

class ResNet34SpectrogramClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet34 = models.resnet34(weights=None)

        # Modify the first layer to accept a single channel input
        self.resnet34.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modify the last layer to have the desired number of output classes
        self.resnet34.fc = nn.Linear(self.resnet34.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet34(x)


def train(dataloader, model, loss, optimizer, scheduler, device, cost):
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, Y) in enumerate(dataloader):
        
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = cost(pred, Y)
        loss.backward()
        optimizer.step()
        
        if batch % 200 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')
            
    scheduler.step()

def test(dataloader, model, device, cost, log = True):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for batch, (X, Y) in enumerate(dataloader):
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            
            test_loss += cost(pred, Y).item()*batch
            correct += (pred.argmax(1)==Y.argmax(1)).type(torch.float).sum().item()
    
    test_loss /= size
    correct /= size

    if log:
        print(f'\nTest Error:\nacc: {(100*correct):>0.1f}%, avg loss: {test_loss:>8f}\n')
    return correct*100


def setup_dataset(train_images = 100, validation_images = 100, test_images = 100, aug = [0,0,0]):
    #load and one-hot encode labels
    print('Loading audio data')
    train_dataset, validation_dataset, test_dataset, labels = load_audio_files('./data/SpeechCommands/speech_commands_v0.02', train_images, validation_images, test_images)
    train_dataset      = [[pad_waveform(X, 16000), one_hot_encode(label, labels)] for [X, label] in train_dataset]
    validation_dataset = [[pad_waveform(X, 16000), one_hot_encode(label, labels)] for [X, label] in validation_dataset]
    test_dataset       = [[pad_waveform(X, 16000), one_hot_encode(label, labels)] for [X, label] in test_dataset]
    
    #before converting into spectrograms for training, we do some data augmentation
    print('Performing waveform augmentations')
    train_dataset_wf_aug = augment_waveforms(train_dataset, noise = aug[0], stft_conv = aug[3])

    print("Computing spectrograms")
    train_dataset = [[waveform_to_spectrogram(waveform), label_vec] for [waveform, label_vec] in train_dataset]
    train_dataset_wf_aug = [[waveform_to_spectrogram(waveform), label_vec] for [waveform, label_vec] in train_dataset_wf_aug]
    validation_dataset = [[waveform_to_spectrogram(waveform), label_vec] for [waveform, label_vec] in validation_dataset]
    test_dataset =  [[waveform_to_spectrogram(waveform), label_vec] for [waveform, label_vec] in test_dataset]
    
    print("Performing spectrogram augmentations")
    train_dataset_spec_aug = augment_spectrogram(train_dataset, spec_aug = aug[1], spec_blur = aug[2])
    
    train_dataset += train_dataset_spec_aug
    train_dataset += train_dataset_wf_aug
    
    return train_dataset, validation_dataset, test_dataset, labels

def full_run(training_images = 100, validation_images = 100, test_images = 100, aug = [0, 0, 0, 0]):
    train_dataset, validation_dataset, test_dataset, labels = setup_dataset(training_images, validation_images, test_images, aug)
    output_channels = len(labels)
        
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        num_workers=12,
        persistent_workers=True,
        shuffle=True
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=32,
        num_workers=12,
        persistent_workers=True,
        shuffle=True
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = ResNet34SpectrogramClassifier(output_channels).to(device)
    
    cost = torch.nn.CrossEntropyLoss()
    learning_rate = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    step_size = 30
    gamma = 0.1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    max_epochs = 100
    best_epoch = 0
    best_acc = 0
    
    for t in range(max_epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        train(train_dataloader, model, cost, optimizer, scheduler, device, cost)
        acc = test(validation_dataloader, model, device, cost, log = False)
        
        if acc > best_acc:
            best_acc = acc
            best_epoch = t
            
        if (t - best_epoch) > 2:
            break
    
        
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        num_workers=12,
        persistent_workers=True,
        shuffle=True
    )
        
    test_acc = test(test_dataloader, model, device, cost, log = True)
            
    return test_acc

def average_runs(training_images = 100, validation_images = 100, test_images = 100, loops = 1, aug = [0, 0, 0, 0]):
    accs = np.array([full_run(training_images, validation_images, test_images, aug) for _ in range(loops)])
    return accs.mean(), accs.std()

if __name__ == '__main__':
    accs = []
    
    plot_augmentations()

    
    aug = [0, 0, 0, 0]
    #a100, s100 = average_runs(100, 100, 200, 3, aug)
    #a300, s300 = average_runs(300, 100, 200, 3, aug)
    #a600, s600 = average_runs(600, 100, 200, 3, aug)
    #a1000, s1000 = average_runs(1000, 100, 200, 3, aug)

        



    pass

    






"""
class ConvNeXtTinySpectrogramClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.convnext = models.convnext.convnext_tiny(weights=None)

        # Modify the first layer to accept a single channel input
        firstconv_output_channels = 96  # From the block_setting of convnext_tiny
        self.convnext.features[0] = models.convnext.Conv2dNormActivation(
            1,
            firstconv_output_channels,
            kernel_size=4,
            stride=4,
            padding=0,
            norm_layer=partial(models.convnext.LayerNorm2d, eps=1e-6),
            activation_layer=None,
            bias=True,
        )

        # Modify the last layer to have the desired number of output classes
        lastconv_output_channels = 768  # From the last CNBlockConfig of convnext_tiny
        self.convnext.classifier[-1] = nn.Linear(lastconv_output_channels, num_classes)

    def forward(self, x):
        return self.convnext(x)

class ConvNeXtSmallSpectrogramClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.convnext = models.convnext.convnext_small(weights=None)

        # Modify the first layer to accept a single channel input
        firstconv_output_channels = 96  # From the block_setting of convnext_small
        self.convnext.features[0] = models.convnext.Conv2dNormActivation(
            1,
            firstconv_output_channels,
            kernel_size=4,
            stride=4,
            padding=0,
            norm_layer=partial(models.convnext.LayerNorm2d, eps=1e-6),
            activation_layer=None,
            bias=True,
        )

        # Modify the last layer to have the desired number of output classes
        lastconv_output_channels = 768  # From the last CNBlockConfig of convnext_tiny
        self.convnext.classifier[-1] = nn.Linear(lastconv_output_channels, num_classes)

    def forward(self, x):
        return self.convnext(x)
"""



