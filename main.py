# -*- coding: utf-8 -*-

import os
import torchaudio
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import models
from torchinfo import summary
import pickle
import torch
from functools import partial
import random


def download_and_save_data():
    default_dir = os.getcwd()
    folder = 'data'
    print(f'Data directory will be: {default_dir}/{folder}')
    
    if os.path.isdir(folder):
        print("Data folder exists.")
    else:
        print("Creating folder.")
        os.mkdir(folder) 
        

def load_audio_files(path: str, max_images):
    #For future, smallest class has 1557 examples
    dataset = []
    
    all_contents = os.listdir(path)
    subdirs = [content for content in all_contents if os.path.isdir(os.path.join(path, content))]
    
    for subdir in subdirs:
        subpath = os.path.join(path, subdir)
        
        walker = random.shuffle([str(p) for p in Path(subpath).glob(f'*.wav')])
        
        for i, file_path in enumerate(walker):
            if (i >= max_images):
                break
            
            waveform, sample_rate = torchaudio.load(file_path)
            dataset.append([waveform, sample_rate, subdir])
            
    return dataset, subdirs

def waveform_to_spectrogram(waveform, sample_rate):
    # Parameters for the spectrogram
    n_fft = 4096        # Increased from 2048
    win_length = None
    hop_length = 256    # Decreased from 512
    n_mels = 256        # Increased from 128
    
    
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
    shifted_log_mel_spectrogram = log_mel_spectrogram - log_mel_spectrogram.min()
    normalized_log_mel_spectrogram = shifted_log_mel_spectrogram / shifted_log_mel_spectrogram.max()

    return normalized_log_mel_spectrogram

def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(spec.squeeze(), origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
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

def add_noise(waveform):
    return waveform + torch.randn(waveform.size())*0.17*waveform.max()
    
def add_spec_aug(spectrogram, F=27, T=10, num_freq_masks=2, num_time_masks=2):
    """
    Apply SpecAugment to a given spectrogram tensor.

    :param spectrogram: Tensor of shape (1, 256, 63) representing a spectrogram
    :param F: Maximum width of the frequency mask
    :param T: Maximum width of the time mask
    :param num_freq_masks: Number of frequency masks to apply
    :param num_time_masks: Number of time masks to apply
    :return: SpecAugmented spectrogram
    """

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

def augment_waveforms(ds, noise = True):
    aug_ds = []
    
    for X, fs, Y in ds:
        if noise:  
            aug_ds.append([add_noise(X), fs, Y])
            
        aug_ds.append([X, fs, Y])
        
    return aug_ds

def augment_spectrogram(ds, spec_aug = True):
    aug_ds = []
    
    for X, Y in ds:
        if spec_aug:  
            aug_ds.append([add_spec_aug(X), Y])
            
        aug_ds.append([X, Y])
        
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



# Create the training function
def train(dataloader, model, loss, optimizer, device, cost):
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


# Create the validation/test function
def test(dataloader, model, device, cost):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for batch, (X, Y) in enumerate(dataloader):
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            
            test_loss += cost(pred, Y).item()*batch
            correct += (pred.argmax(1)==Y.argmax(1)).type(torch.float).sum().item()
    
    test_loss /= size/2
    correct /= size

    print(f'\nTest Error:\nacc: {(100*correct):>0.1f}%, avg loss: {test_loss:>8f}\n')
    return correct*100


def setup_dataset(aug_noise = True, aug_spec_aug = True, max_images = 100):
    #load and one-hot encode labels
    print('Loading audio data')
    dataset, labels = load_audio_files('./data/SpeechCommands/speech_commands_v0.02', max_images)
    dataset = [[pad_waveform(X, 16000), fs, one_hot_encode(label, labels)] for [X, fs, label] in dataset]

    

    #split into training and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    #before converting into spectrograms for training, we do some data augmentation
    print('Performing waveform augmentations')
    train_dataset = augment_waveforms(train_dataset, noise = aug_noise)

    wf = train_dataset[0][0]

    print("Computing spectrograms")
    train_dataset = [[waveform_to_spectrogram(waveform, fs), label_vec] for [waveform, fs, label_vec] in train_dataset]
    test_dataset =  [[waveform_to_spectrogram(waveform, fs), label_vec] for [waveform, fs, label_vec] in test_dataset]
    
    print("Performing spectrogram augmentations")
    train_dataset = augment_spectrogram(train_dataset, spec_aug = aug_spec_aug)
    
    return train_dataset, test_dataset, labels

def setup_dataloaders(aug_noise = True, aug_spec_aug = True, max_images = 1500):
    train_dataset, test_dataset, labels = setup_dataset(aug_noise, aug_spec_aug, max_images)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        num_workers=12,
        persistent_workers=True,
        shuffle=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        num_workers=12,
        persistent_workers=True,
        shuffle=True
    )
    
    return train_dataloader, test_dataloader, labels

def run_train_test(train_dataloader, test_dataloader, labels, arch = 'small'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    
    model = ConvNeXtSmallSpectrogramClassifier(len(labels)).to(device)
    if arch == 'tiny':
        model = ConvNeXtTinySpectrogramClassifier(len(labels)).to(device)
    elif arch == 'resnet34':
        model = ResNet34SpectrogramClassifier(len(labels)).to(device)
    
    cost = torch.nn.CrossEntropyLoss()
    learning_rate = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    max_epochs = 100
    best_epoch = 0
    best_acc = 0
    
    for t in range(max_epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        train(train_dataloader, model, cost, optimizer, device, cost)
        acc = test(test_dataloader, model, device, cost)
        
        if acc > best_acc:
            best_acc = acc
            best_epoch = t
            
        if (t - best_epoch) > 3:
            break
        
    print('BEST:')
    print(best_acc)
    print('Done!')
    return best_acc
    
def full_run(max_images, aug_noise, aug_spec_aug, arch, loops = 1):
    train_dataloader, test_dataloader, labels = setup_dataloaders(max_images = max_images, aug_noise = aug_noise, aug_spec_aug = aug_spec_aug)
    
    
    avg_acc = 0
    best_acc = 0
    for _ in range(loops):    
        acc = run_train_test(train_dataloader, test_dataloader, labels, arch)
        avg_acc += acc/loops
        
        if acc > best_acc:
            best_acc = acc
            
    print('Finished full run with max images {}, noise {}, spec_aug {}, arch{}'.format(max_images, aug_noise, aug_spec_aug, arch))
    print(avg_acc)
    print(best_acc)
            
    return avg_acc, best_acc

if __name__ == '__main__':
    accuracies = []
    
    #acc_tiny, best_acc_tiny = full_run(1500, True, True, 'tiny', 2)
    #acc_small, best_acc_small = full_run(1500, True, True, 'small', 2)
    
    """
    limits = np.array(list(range(1, 8)))*200
    iters = 3
    
    for limit in limits:
        vec = []
        

        
        vec.append(acc_no)
        vec.append(acc_noise)
        vec.append(acc_spec)
        accuracies.append(vec)
    
    print(accuracies)"""
    
    pass

    
    
"""
SUMMARY OF RESULTS:
    
    Images  Noise  SpecAug  Architecture     Accuracy  
    1500    Y      Y        ResNet34         92.2%
    500     Y      Y        ResNet34         88.9%
    200     Y      Y        ResNet34         84.5%
    100     Y      Y        ResNet34         79.0%




    1500    Y      Y        ConvNext-tiny    90.5%
    1500    Y      Y        ConvNext-small   90.5%
    100     Y      Y        ConvNeXt-tiny    67.6%
    100     Y      Y        ConvNeXt-small   67.4%    
    
        

"""
    


#92.2 % best so far, ResNet34

