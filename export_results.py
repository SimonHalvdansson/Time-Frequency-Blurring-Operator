#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

accs = [[np.array(1), np.array(2)],
        [np.array(3), np.array(3)]]

def format_acc(ac, sd):
    return '{:.1f} \pm {:.1f}'.format(ac, sd)    

CONFIG_COUNTS = [1,2]
CONFIG_AUGS = [1,2]

def export_results(accs):
    aug_names = ['None', 'White noise', 'SpecAugment', 'STFT-blur', 'SpecBlur', 'White noise + SpecAug', 'STFT-blur + SpecBlur', 'All']
    
    for aug_ind in range(len(CONFIG_AUGS)):
        s = aug_names[aug_ind]
        for count_ind in range(len(CONFIG_COUNTS)):
            avg, std = accs[count_ind][aug_ind].mean(), accs[count_ind][aug_ind].std()
            
            s += ' & '
            s += format_acc(avg, std)
            
        print(s)
            

export_results(accs)