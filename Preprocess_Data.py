# -*- coding: utf-8 -*-
"""
Created on Wed Sept  7 16:32 2022

@author: antho
"""

#%% Import Libraries
import mne
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.io import savemat

#%% Link to Dataset

# https://figshare.com/articles/dataset/EEG_Data_New/4244171

#%% Load, Normalize, and Segment Data (Assumes that HC and MDD Data are in Separate Directories)

folderpath1 = "C:/Users/cellis42/Documents/AnthonyFiles/Calhoun_Lab/Projects/Spectral_Explainability/MDD/RawData/HC/"
folderpath2 = "C:/Users/cellis42/Documents/AnthonyFiles/Calhoun_Lab/Projects/Spectral_Explainability/MDD/RawData/MDD/"

target_fs = 100

for f in range(0,2):
    if f == 0:
        folderpath = folderpath1
    else:
        folderpath = folderpath2
        
    # Find Files
    files = os.listdir(folderpath)
    
    # Load Files
    subject = []
    data = []
    for i in range(0,len(files)):
        file=mne.io.read_raw_edf(os.path.join(folderpath,files[i])).resample(sfreq=target_fs)
        data_len = np.shape(file.times)[0]
        nepochs = np.int(np.floor(data_len/(target_fs*30))-1)
        events = mne.make_fixed_length_events(file, start=0, stop=nepochs*30-1, duration=2.5) # make events every 2.5 seconds
        nepochs = np.shape(events)[0]
        # epoch_file =  mne.Epochs(file, events, tmin=0, tmax=5,baseline=None)
        epoch_file =  mne.Epochs(file, events, tmin=-15.0, tmax=15.0,baseline=None) # When making epochs, define them as 12.5 seconds before to 12.5 seconds after previously defined events
        file = []
        
        df=epoch_file.to_data_frame()
        epoch_file = []
        channels = df.columns[3:]
        channels_to_use = ['EEG Fp1-LE', 'EEG Fp2-LE', 'EEG F7-LE', 'EEG F3-LE', 'EEG Fz-LE', 'EEG F4-LE', 'EEG F8-LE', 'EEG T3-LE', 'EEG C3-LE', 'EEG Cz-LE', 'EEG C4-LE', 'EEG T4-LE', 'EEG T5-LE', 'EEG P3-LE', 'EEG Pz-LE', 'EEG P4-LE', 'EEG T6-LE', 'EEG O1-LE', 'EEG O2-LE']
        
        epochs = df.epoch
        df=df[channels_to_use]
    
        # Z-Score Each Channel
        mean_vals = df.mean(axis=0)
        sd_vals = df.std(axis=0)
        for ch in channels_to_use:
            df[ch] -= mean_vals[ch]
            df[ch] /= sd_vals[ch]
        
        if f == 0:
            subj = int(int(files[i][3:5])+f*100) # NC subject numbers are their actual numbers, MDD start from 100 + their subject number
        else:
            subj = int(int(files[i][5:7])+f*100) # NC subject numbers are their actual numbers, MDD start from 100 + their subject number

        count = 0
        for epoch in range(np.min(epochs),np.max(epochs)+1):
            vals = np.array(df.iloc[list(np.arange(len(epochs))[list(epoch*np.ones_like(epochs)==epochs)])]).transpose()[None]
            vals = vals[...,:30*target_fs]
            if i + epoch == np.min(epochs):
                data = list(vals)
            else:
                data = np.append(data,vals,axis=0)
            count+=1
        
        df = []
        
        if i == 0:
            subject = list((subj*np.ones((count,))).astype(int))
        else:
            subject.extend(list((subj*np.ones((count,))).astype(int)))
        
        print(i)
        
    if f == 0:
        # data_out = data
        # subject_out = subject
        label = np.zeros_like(subject)
        filename1 = 'C:/Users/cellis42/Documents/AnthonyFiles/Calhoun_Lab/Projects/Spectral_Explainability/Pretraining/segmented_hc1_data_like_sleep'
        filename2 = 'C:/Users/cellis42/Documents/AnthonyFiles/Calhoun_Lab/Projects/Spectral_Explainability/Pretraining/segmented_hc2_data_like_sleep'

    else:
        # data_out = np.concatenate((data_out,data),axis=0)
        # subject_out = np.concatenate((subject_out,subject),axis=0)
        label = np.ones_like(subject)
        filename1 = 'C:/Users/cellis42/Documents/AnthonyFiles/Calhoun_Lab/Projects/Spectral_Explainability/Pretraining/segmented_mdd1_data_like_sleep'
        filename2 = 'C:/Users/cellis42/Documents/AnthonyFiles/Calhoun_Lab/Projects/Spectral_Explainability/Pretraining/segmented_mdd2_data_like_sleep'
    
    # Split Data into Separate Files for Saving
    n_samples_per_file = np.int(np.floor(len(label)/2))
    
    save_data1 = {'data':data[:n_samples_per_file,...],'subject':subject[:n_samples_per_file],'channels':channels_to_use,'label':label[:n_samples_per_file]}

    np.save(filename1,save_data1)
    
    save_data1 = [];
    
    save_data2 = {'data':data[n_samples_per_file:,...],'subject':subject[n_samples_per_file:],'channels':channels_to_use,'label':label[n_samples_per_file:]}

    np.save(filename2,save_data2)
    
    save_data2 = [];