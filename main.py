# -*- coding: utf-8 -*-

import os
import torch
import torchaudio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
import pickle
from tqdm import tqdm
from tinyvit import TinyViT
from matplotlib.gridspec import GridSpec


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
    
    for subdir in tqdm(subdirs, desc="Loading Audio Files"):
        
        subpath = os.path.join(path, subdir)
        
        walker = [str(p) for p in Path(subpath).glob('*.wav')]
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

def waveform_to_spectrogram(waveform, n_fft = 256, win_length = None, hop_length = 128):
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
    

def waveform_to_log_mel_spectrogram(waveform):
    n_fft = 4096
    win_length = None
    hop_length = 256
    n_mels = 256
    
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
    
    waveform = waveform.reshape(1, -1)  # Make sure the input tensor has the correct shape
    mel_spectrogram = mel_spectrogram_transform(waveform)
    
    log_mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
    
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

def plot_spec_blur():
    # Plots spectrogram and log-mel spectrogram before/after SpecBlur is applied, for use in a paper
    ds, _, _, labels = load_audio_files('./data/SpeechCommands/speech_commands_v0.02', 1, 0, 0)
    wf = ds[0][0]
    wf = pad_waveform(wf, 16000)
    
    n_fft = 4096
    hop_length = 256
    n_mels = 256
    
    spec_og = waveform_to_spectrogram(wf, n_fft, hop_length)
    
    f_max = 10000
    max_bin = int(f_max * n_fft / 16000)
    
    spec_og = spec_og[:, :max_bin, :]
    
    # Apply SpecBlur to the original spectrogram
    spec_og_blur = add_spec_blur(spec_og)
    
    # Convert to mel scale
    mel_rescale = torchaudio.transforms.MelScale(n_mels, 16000, 0.0, None, n_fft // 2 + 1, 'slaney')
    spec_og_lm = mel_rescale(spec_og)
    spec_og_blur_lm = mel_rescale(spec_og_blur)
    
    # List for easy indexing in the loop
    spectrograms = [spec_og.numpy(), spec_og_blur.numpy(), spec_og_lm.numpy(), spec_og_blur_lm.numpy()]
    titles = [
        'Original Spectrogram (dB)', 'Blurred Spectrogram (dB)', 
        'Mel Rescaled Original Spectrogram (dB)', 'Mel Rescaled Blurred Spectrogram (dB)'
    ]
    
    # Create a 2x2 grid for the plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), dpi=300)

    # Adjust spacing if needed
    fig.subplots_adjust(hspace=0.4)  # You can adjust this value as needed

    # Use nested loop for row and column indexing
    for i, ax in enumerate(axes.flatten()):
        extent = [0, 1, 0, f_max] if i < 2 else [0, 1, 0, n_mels]
        ax.imshow(spectrograms[i].squeeze(), aspect="auto", origin="lower", extent=extent)
        ax.set_title(titles[i])
        ax.set_xlabel('Time [s]')
        if i % 2 == 0:  # Set y-label for the left column
            ax.set_ylabel('Frequency [Hz]' if i == 0 else 'Mel frequency bin')
    
    plt.show()
    

def plot_stft_conv(save=False, transparent = False):
    # Load the dataset and prepare data
    ds, _, _, labels = load_audio_files('./data/SpeechCommands/speech_commands_v0.02', 1, 0, 0)
    wf = ds[0][0]
    wf = pad_waveform(wf, 16000)
    wf_e = wf.pow(2).sum()
        
    wf_stft = add_stft_conv(wf, sigma_time=1.2, sigma_freq=6)
    
    wf_stft_e = wf_stft.pow(2).sum()
    wf_stft = wf_stft * np.sqrt((wf_e / wf_stft_e))
    
    if save:
        torchaudio.save('before.wav', wf, 16000)
        torchaudio.save('after.wav', wf_stft, 16000)

    n_fft = 4096
    hop_length = 256
    
    s1 = waveform_to_spectrogram(wf, n_fft, hop_length)
    s2 = waveform_to_spectrogram(wf_stft, n_fft, hop_length)
    s3 = waveform_to_log_mel_spectrogram(wf)
    s4 = waveform_to_log_mel_spectrogram(wf_stft)
    
    f_max = 10000
    max_bin = int(f_max * n_fft / 16000)
    
    s1 = s1[:, :max_bin, :]
    s2 = s2[:, :max_bin, :]
    
    # Create a 2x2 grid for the plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), dpi=300)

    # Adjust spacing
    fig.subplots_adjust(hspace=0.3)  # Increase vertical space between rows

    # Plot original spectrogram in top-left
    axes[0, 0].imshow(s1.numpy().squeeze(), aspect="auto", origin="lower", extent=[0, 1, 0, f_max])
    axes[0, 0].set_title('Original spectrogram (dB)')
    axes[0, 0].set_xlabel('Time [s]')
    axes[0, 0].set_ylabel('Frequency [Hz]')

    # Plot STFT-conv spectrogram in top-right
    axes[0, 1].imshow(s2.numpy().squeeze(), aspect="auto", origin="lower", extent=[0, 1, 0, f_max])
    axes[0, 1].set_title('STFT-conv spectrogram (dB)')
    axes[0, 1].set_xlabel('Time [s]')

    # Plot original log-mel spectrogram in bottom-left
    axes[1, 0].imshow(s3.numpy().squeeze(), aspect="auto", origin="lower", extent=[0, 1, 0, 256])
    axes[1, 0].set_title('Original log-mel spectrogram (dB)')
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Mel frequency bin')

    # Plot STFT-conv log-mel spectrogram in bottom-right
    axes[1, 1].imshow(s4.numpy().squeeze(), aspect="auto", origin="lower", extent=[0, 1, 0, 256])
    axes[1, 1].set_title('STFT-conv log-mel spectrogram (dB)')
    axes[1, 1].set_xlabel('Time [s]')

    if save:
        plt.savefig('combined_spectrograms.png', transparent = transparent)  # Save a single file with all plots
    plt.show()


def plot_augmentations():
    # Plots the summary of augmentation methods presented in the paper
    ds, _, _, labels = load_audio_files('./data/SpeechCommands/speech_commands_v0.02', 1, 0, 0)
    wf = ds[0][0]
    wf = pad_waveform(wf, 16000)
    
    wf_noise = add_noise(wf)
    wf_stft_conv = add_stft_conv(wf)
    
    spec = waveform_to_log_mel_spectrogram(wf)
    spec_noise = waveform_to_log_mel_spectrogram(wf_noise)
    spec_stft_conv = waveform_to_log_mel_spectrogram(wf_stft_conv)
    spec_aug = add_spec_aug(spec)
    spec_blur = add_spec_blur(spec)
        
    spectrograms = [spec, spec_noise, spec_aug, spec_stft_conv, spec_blur]
    titles = ['Original', 'White noise', 'SpecAugment', 'STFT convolved', 'SpecBlur']
    
    fig = plt.figure(figsize=(15, 8), dpi=300)
    gs = GridSpec(2, 6, figure=fig)
    
    # Top row
    ax0 = fig.add_subplot(gs[0, 0:2])
    ax1 = fig.add_subplot(gs[0, 2:4])
    ax2 = fig.add_subplot(gs[0, 4::])

    # Bottom row, centered
    ax3 = fig.add_subplot(gs[1, 1:3])
    ax4 = fig.add_subplot(gs[1, 3:5])
    
    axes = [ax0, ax1, ax2, ax3, ax4]

    for i, ax in enumerate(axes):
        ax.imshow(spectrograms[i].numpy().squeeze(), aspect="auto", origin="lower", extent=[0, 1, 0, 256])
        ax.set_title(titles[i])
        if i >= 3:  # Bottom row
            ax.set_xlabel('Time [s]')
        if i % 3 == 0 or i == 3:  # Leftmost columns
            ax.set_ylabel('Mel frequency bin')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)  # Adjust vertical spacing as needed

    plt.savefig('augs.png', transparent=True)

    plt.show()

        
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
    
    for X, Y in tqdm(ds, desc="Augmenting waveforms", leave=False):
        if noise:  
            aug_ds.append([add_noise(X), Y])
        if stft_conv:
            aug_ds.append([add_stft_conv(X), Y])
        
    return aug_ds

def augment_spectrogram(ds, spec_aug = False, spec_blur = False):
    aug_ds = []
    
    for X, Y in tqdm(ds, desc="Augmenting spectrograms", leave=False):
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

class TinyViTWrapper(nn.Module):
    def __init__(self, num_classes, param_mode='11m'):
        super(TinyViTWrapper, self).__init__()
        self.img_size = 224
        
        model_kwargs_5m = dict(
            embed_dims=[64, 128, 160, 320],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            drop_path_rate=0.0,
        )
        
        model_kwargs_11m = dict(
            embed_dims=[64, 128, 256, 448],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 8, 14],
            window_sizes=[7, 7, 14, 7],
            drop_path_rate=0.1,
        )
        
        model_kwargs_21m = dict(
            embed_dims=[96, 192, 384, 576],
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 18],
            window_sizes=[7, 7, 14, 7],
            drop_path_rate=0.2,
        )
        
        model_kwargs = None
        
        if (param_mode == '5m'):
            model_kwargs = model_kwargs_5m
        elif (param_mode == '11m'):
            model_kwargs = model_kwargs_11m
        else:
            model_kwargs = model_kwargs_21m
                
        self.tiny_vit = TinyViT(img_size=self.img_size, 
                                in_chans=1,
                                num_classes=num_classes,
                                **model_kwargs)

    def forward(self, x):
        # Resize image to square (self.img_size x self.img_size)
        x = torch.nn.functional.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)

        # Pass the resized image through TinyViT
        x = self.tiny_vit(x)
        return x


def train(dataloader, model, loss, optimizer, scheduler, device, cost):
    model.train()
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch, (X, Y) in enumerate(progress_bar):
        
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = cost(pred, Y)
        loss.backward()
        optimizer.step()
        
        progress_bar.set_description(f"Training - Loss: {loss.item():.4f}")
            
    scheduler.step()
    return loss.item()

def test(dataloader, model, device, cost, log = True):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for batch, (X, Y) in enumerate(tqdm(dataloader, desc="Validating/Testing", leave=False)):
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
    train_dataset, validation_dataset, test_dataset, labels = load_audio_files('./data/SpeechCommands/speech_commands_v0.02', train_images, validation_images, test_images)
    train_dataset      = [[pad_waveform(X, 16000), one_hot_encode(label, labels)] for [X, label] in train_dataset]
    validation_dataset = [[pad_waveform(X, 16000), one_hot_encode(label, labels)] for [X, label] in validation_dataset]
    test_dataset       = [[pad_waveform(X, 16000), one_hot_encode(label, labels)] for [X, label] in test_dataset]
    
    #before converting into spectrograms for training, we do some data augmentation
    train_dataset_wf_aug = augment_waveforms(train_dataset, noise = aug[0], stft_conv = aug[3])


    if train_dataset:
        train_dataset = [[waveform_to_log_mel_spectrogram(waveform), label_vec] 
                         for waveform, label_vec in tqdm(train_dataset, desc="Processing Train Dataset")]
    
    if train_dataset_wf_aug:
        train_dataset_wf_aug = [[waveform_to_log_mel_spectrogram(waveform), label_vec] 
                                for waveform, label_vec in tqdm(train_dataset_wf_aug, desc="Processing Augmented Train Dataset")]
    
    if validation_dataset:
        validation_dataset = [[waveform_to_log_mel_spectrogram(waveform), label_vec] 
                              for waveform, label_vec in tqdm(validation_dataset, desc="Processing Validation Dataset")]
    
    if test_dataset:
        test_dataset = [[waveform_to_log_mel_spectrogram(waveform), label_vec] 
                        for waveform, label_vec in tqdm(test_dataset, desc="Processing Test Dataset")]

    
    train_dataset_spec_aug = augment_spectrogram(train_dataset, spec_aug = aug[1], spec_blur = aug[2])
    
    train_dataset += train_dataset_spec_aug
    train_dataset += train_dataset_wf_aug
    
    return train_dataset, validation_dataset, test_dataset, labels

def full_run(training_images = 100, validation_images = 100, test_images = 100, aug = [0, 0, 0, 0]):
    train_dataset, validation_dataset, test_dataset, labels = setup_dataset(training_images, validation_images, test_images, aug)
    output_channels = len(labels)
        
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        shuffle=True
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        shuffle=True
    )
    
    device = 'cpu'
    
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    
    model = ResNet34SpectrogramClassifier(output_channels).to(device)
        
    if NET_TYPE == 'vit':
        model = TinyViTWrapper(output_channels).to(device)
            
    cost = torch.nn.CrossEntropyLoss()
    learning_rate = LEARNING_RATE
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    step_size = 30
    gamma = 0.1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    max_epochs = 100
    best_epoch = 0
    best_acc = 0
    
    for t in range(max_epochs):
        print(f'\nEpoch {t+1}\n-------------------------------')
        loss = train(train_dataloader, model, cost, optimizer, scheduler, device, cost)
        acc = test(validation_dataloader, model, device, cost, log = False)
        
        print("Final loss: {:.4f}".format(loss))
        print("Validation accuracy: {:.2f}%".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = t
            
        if (t - best_epoch) > 2:
            break
    
    print('Training done, computing test performance')
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        shuffle=True
    )
        
    test_acc = test(test_dataloader, model, device, cost, log = True)
            
    return test_acc

def average_runs(training_images = 100, validation_images = 100, test_images = 100, loops = 1, aug = [0, 0, 0, 0]):
    t0 = time.time()
    accs = []
    for i in range(loops):
        print('Starting run {}/{} with augmentation'.format(i+1, loops))
        print(aug)
        accs.append(full_run(training_images, validation_images, test_images, aug))
        
    accs = np.array(accs)
    dt = time.time() - t0
    
    return accs.mean(), accs.std(), dt/60

def format_acc(ac, se):
    return '{:.2f} \pm {:.2f}'.format(ac, se)

def find_suitable_index(accs):   
    #First check if there's an index with too little data
    for count_ind in range(len(CONFIG_COUNTS)):
        for aug_ind in range(len(CONFIG_AUGS)):
            if len(accs[count_ind][aug_ind]) < COUNT_LIMIT:
                print('Config with count: {} and aug: {} had < {} data, starting training'
                      .format(CONFIG_COUNTS[count_ind], CONFIG_AUGS[aug_ind], COUNT_LIMIT))
                return count_ind, aug_ind
        
        
    #new we check where we have the largest standard error (std/sqrt(n)) because this is the error of the esimator
    se_max = 0
    se_max_count_ind = None
    se_max_aug_ind = None
    se_sum = 0
    
    for count_ind in range(len(CONFIG_COUNTS)):
        for aug_ind in range(len(CONFIG_AUGS)):
            n = len(accs[count_ind][aug_ind])
            #we do the derivative instead because we want to decrease se as much as possible
            se = accs[count_ind][aug_ind].std()/(n**1.5)/CONFIG_COUNTS[count_ind]
            se_sum += se*n*CONFIG_COUNTS[count_ind]
            
            if se > se_max:
                se_max = se
                se_max_count_ind = count_ind
                se_max_aug_ind = aug_ind
                
    
                
    print('Maximum SE config, count: {}, aug: {}, starting training'
              .format(CONFIG_COUNTS[se_max_count_ind], CONFIG_AUGS[se_max_aug_ind]))
    print('Current SE sum: {:.2f}'.format(se_sum))
    
    return se_max_count_ind, se_max_aug_ind

    

def load_accs():
    file = open('accs_' + NET_TYPE, 'rb')
    res = pickle.load(file)
    file.close()
    
    return res

def save_accs(res, net='resnet'):
    file = open('accs_' + NET_TYPE, 'wb')
    pickle.dump(res, file)
    file.close()

def improve(report_accs = True, new = False):
    if new:
        res = []
        for count in CONFIG_COUNTS:
            count_results = []
            for aug in CONFIG_AUGS:              
                count_results.append(np.array([]))
            
            res.append(count_results)
        save_accs(res)
    
    if report_accs:
        print('Starting improve run, starting setup:')
        report_acc_stats()
        
    
    while True:
        accs = load_accs()
        
        conf_count_ind, conf_aug_ind = find_suitable_index(accs)
        count = CONFIG_COUNTS[conf_count_ind]
        aug = CONFIG_AUGS[conf_aug_ind]
        
        acc = full_run(round(count * 0.8), round(count * 0.2), 200, aug)
        accs[conf_count_ind][conf_aug_ind] = np.append(accs[conf_count_ind][conf_aug_ind], acc)
        print('Added test accuracy {:.2f}% to count: {}, aug: {}'.format(acc, count, aug))
        
        if report_accs:
            report_acc_stats(accs)
        
        save_accs(accs)
    
def export_results(accs = None):
    if accs == None:
        accs = load_accs()
        
    print('')
        
    aug_names = ['None', 'White noise', 'SpecAugment', 'STFT-blur', 'SpecBlur', 'White noise + SpecAug', 'STFT-blur + SpecBlur', 'All']
    #               0           1             2             3           4                   5                      6               7
    
    se_sum = 0
    
    for aug_ind in range(len(CONFIG_AUGS)):
        s = aug_names[aug_ind]
        for count_ind in range(len(CONFIG_COUNTS)):
            samples = accs[count_ind][aug_ind]
            
            avg, se = samples.mean(), samples.std()/np.sqrt(len(samples))
            
            se_sum += se
            
            s += ' & $'
            s += format_acc(avg, se)
            s += '$'
            if count_ind == len(CONFIG_COUNTS)-1:
                s += '\\\\'
            
        print(s)
        
    print('')
    print('Sum of standard errors: {:.2f}'.format(se_sum))
    
    means = [ [ accs[count_ind][aug_ind].mean() for aug_ind in range(len(CONFIG_AUGS)) ] for count_ind in range(len(CONFIG_COUNTS))]
    ses = [ [ accs[count_ind][aug_ind].std()/np.sqrt(len(accs[count_ind][aug_ind])) for aug_ind in range(len(CONFIG_AUGS)) ] for count_ind in range(len(CONFIG_COUNTS))]
    
    comps = [[3,0], [4,0], [5,7], [3,4]]
    
    for comp in comps:
        i1 = comp[0]
        i2 = comp[1]
        
        type1 = aug_names[i1]
        type2 = aug_names[i2]
        print('')
        
        print('{} vs {}:'.format(type1, type2))
        for count_ind in range(len(CONFIG_COUNTS)):
            normalized_diff = np.abs(means[count_ind][i1] - means[count_ind][i2])/np.sqrt(ses[count_ind][i1]**2 + ses[count_ind][i2]**2)
            
            print('Count: {}, {:.2f} σ'.format(CONFIG_COUNTS[count_ind], normalized_diff))

def report_acc_stats(accs = None):
    if accs == None:
        accs = load_accs()
    
    counts = [ [ len(accs[count_ind][aug_ind]) for aug_ind in range(len(CONFIG_AUGS)) ] for count_ind in range(len(CONFIG_COUNTS))]
    print('')
    print('Counts for each config: ')
    print(counts)
    print('')
    print('Total number of runs: ' + str(sum(sum(np.array(counts)))))
    print('')
    
    if np.array(counts).min() > 0:
        print('Standard error for each config')
        ses = [ [ '{:.2f}'.format(accs[count_ind][aug_ind].std()/np.sqrt(len(accs[count_ind][aug_ind]))) for aug_ind in range(len(CONFIG_AUGS)) ] for count_ind in range(len(CONFIG_COUNTS))]
        print(ses)
    

    
CONFIG_COUNTS = [100, 300, 600, 1000]

CONFIG_AUGS = [[0,0,0,0],  #none
        [1,0,0,0],  #white noise
        [0,1,0,0],  #spec augment
        [0,0,1,0],  #stft conv
        [0,0,0,1],  #specblur
        [1,1,0,0],  #white noise + specaug
        [0,0,1,1],  #stft conv + specblur
        [1,1,1,1]]  #everything

NET_TYPE = 'resnet'
NET_TYPE = 'vit'
NUM_WORKERS = 8
COUNT_LIMIT = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

if __name__ == '__main__':   
    #plot_augmentations()
    plot_stft_conv(True, True)
    #plot_spec_blur()
    
    #improve()
    #export_results()
    #report_acc_stats()
    pass

    



















