import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'util'))
import numpy as np
import argparse
import h5py
import librosa
from scipy import signal
import matplotlib.pyplot as plt
import time
import math
import pandas as pd
import random
import torch

from utils import (create_folder, read_audio, pad_truncate_sequence, read_metadata)
import config

class LogMelExtractor(object):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax):
        '''Log mel feature extractor. 
        
        Args:
          sample_rate: int
          window_size: int
          hop_size: int
          mel_bins: int
          fmin: int, minimum frequency of mel filter banks
          fmax: int, maximum frequency of mel filter banks
        '''
        
        self.window_size = window_size
        self.hop_size = hop_size
        self.window_func = np.hanning(window_size)
        
        self.melW = librosa.filters.mel(
            sr=sample_rate, 
            n_fft=window_size, 
            n_mels=mel_bins, 
            fmin=fmin, 
            fmax=fmax).T
        '''(n_fft // 2 + 1, mel_bins)'''

    def transform(self, audio):
        '''Extract feature of a singlechannel audio file. 
        
        Args:
          audio: (samples,)
          
        Returns:
          feature: (frames_num, freq_bins)
        '''
    
        window_size = self.window_size
        hop_size = self.hop_size
        window_func = self.window_func
        
        # Compute short-time Fourier transform
        stft_matrix = librosa.core.stft(
            y=audio, 
            n_fft=window_size, 
            hop_length=hop_size, 
            window=window_func, 
            center=True, 
            dtype=np.complex64, 
            pad_mode='reflect').T
        '''(N, n_fft // 2 + 1)'''
    
        # Mel spectrogram
        mel_spectrogram = np.dot(np.abs(stft_matrix) ** 2, self.melW)
        
        # Log mel spectrogram
        logmel_spectrogram = librosa.core.power_to_db(
            mel_spectrogram, ref=1.0, amin=1e-10, 
            top_db=None)
        
        logmel_spectrogram = logmel_spectrogram.astype(np.float32)
        
        return logmel_spectrogram

def deltas(X_in):
    X_out = (X_in[:,2:]-X_in[:,:-2])/10.0
    X_out = X_out[:,1:-1]+(X_in[:,4:]-X_in[:,:-4])/5.0
    out = np.zeros((X_in.shape[0], X_in.shape[1]))
    out[:,2:-2] = X_out
    return out

def MaxMinNormalization(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x
    
def calculate_feature_for_all_audio_files(args):
    '''Calculate feature of audio files and write out features to a hdf5 file. 
    
    Args:
      dataset_dir: string
      workspace: string
      mini_data: bool, set True for debugging on a small part of data
    '''

    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    mini_data = args.mini_data
    
    sample_rate = config.sample_rate
    window_size = config.window_size
    hop_size = config.hop_size
    mel_bins = config.mel_bins
    fmin = config.fmin
    fmax = config.fmax
    frames_per_second = config.frames_per_second
    frames_num = config.frames_num
    total_samples = config.total_samples
    lb_to_idx = config.lb_to_idx
    audio_duration_clip = config.audio_duration_clip
    audio_stride_clip = config.audio_stride_clip
    audio_duration = config.audio_duration
    audio_num = config.audio_num
    total_frames = config.total_frames
    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
        
    audios_dir = os.path.join(dataset_dir, 'audio')
    metadata_path = os.path.join(dataset_dir, 'meta', 'esc50.csv')
    feature_path = os.path.join(workspace, 'features', 
        '{}logmel_{}frames_{}melbins.h5'.format(prefix, frames_per_second, mel_bins))
    create_folder(os.path.dirname(feature_path))    
    # Feature extractor
    feature_extractor = LogMelExtractor(
        sample_rate=sample_rate, 
        window_size=window_size, 
        hop_size=hop_size, 
        mel_bins=mel_bins, 
        fmin=fmin, 
        fmax=fmax)

    # Read metadata
    meta_dict = read_metadata(metadata_path)

    # Extract features and targets 
    if mini_data:
        mini_num = 10
        total_num = len(meta_dict['filename'])
        random_state = np.random.RandomState(1234)
        indexes = random_state.choice(total_num, size=mini_num, replace=False)
        for key in meta_dict.keys():
            meta_dict[key] = meta_dict[key][indexes]
        
    print('Extracting features of all audio files ...')
    extract_time = time.time()
    # Hdf5 file for storing features and targets
    hf = h5py.File(feature_path, 'w')

    hf.create_dataset(
        name='filename', 
        data=[filename.encode() for filename in meta_dict['filename']], 
        dtype='S80')

    if 'fold' in meta_dict.keys():
        hf.create_dataset(
            name='fold', 
            data=[fold for fold in meta_dict['fold']], 
            dtype=np.int64)
            
    if 'target' in meta_dict.keys():
        hf.create_dataset(
            name='target', 
            data=[target for target in meta_dict['target']], 
            dtype=np.int64)
            
    if 'category' in meta_dict.keys():
        hf.create_dataset(
            name='category', 
            data=[category.encode() for category in meta_dict['category']], 
            dtype='S80')
    if 'esc10' in meta_dict.keys():
        hf.create_dataset(
            name='esc10', 
            data=[esc10 for esc10 in meta_dict['esc10']], 
            dtype=np.bool)
    if 'src_file' in meta_dict.keys():
        hf.create_dataset(
            name='src_file', 
            data=[src_file for src_file in meta_dict['src_file']], 
            dtype=np.int64)
    if 'take' in meta_dict.keys():
        hf.create_dataset(
            name='take', 
            data=[take.encode() for take in meta_dict['take']], 
            dtype='S24')
    

    hf.create_dataset(
        name='feature', 
        shape=(0, audio_num, 3, frames_num, mel_bins), 
        maxshape=(None, audio_num, 3, frames_num, mel_bins), 
        dtype=np.float32)

    for (n, filename) in enumerate(meta_dict['filename']):
        audio_path = os.path.join(audios_dir, filename)
        print(n, audio_path)
        
        # Read audio
        (audio, _) = read_audio(
            audio_path=audio_path, 
            target_fs=sample_rate)
        
        # Pad or truncate audio recording to the same length
        audio = pad_truncate_sequence(audio, total_samples)
        # Extract feature
        fea_list = []
#         for i in range(audio_num):
#             audio_clip = audio[i*sample_rate*audio_stride_clip: (i+2)*sample_rate*audio_stride_clip]
#             feature = feature_extractor.transform(audio_clip)
#             feature = feature[0 : frames_per_second*audio_duration_clip]
#             fea_list.append(feature)
        feature = feature_extractor.transform(audio)
#         # Remove the extra log mel spectrogram frames caused by padding zero
        feature = feature[0 : total_frames]
        feature = MaxMinNormalization(feature)
        feature_delta = deltas(feature)
        feature_delta = MaxMinNormalization(feature_delta)
        feature_delta2 = deltas(feature_delta)
        feature_delta2 = MaxMinNormalization(feature_delta2)
        for i in range(audio_num):
            feature_clip = feature[i*frames_per_second*audio_stride_clip: (i+audio_duration_clip)*frames_per_second*audio_stride_clip]
            feature_delta_clip = feature_delta[i*frames_per_second*audio_stride_clip: (i+audio_duration_clip)*frames_per_second*audio_stride_clip]
            feature_delta2_clip = feature_delta2[i*frames_per_second*audio_stride_clip: (i+audio_duration_clip)*frames_per_second*audio_stride_clip]
            feature_clip = feature_clip[None, :, :]
            feature_delta_clip = feature_delta_clip[None, :, :]
            feature_delta2_clip = feature_delta2_clip[None, :, :]
            f = np.concatenate((feature_clip, feature_delta_clip, feature_delta2_clip), 0)
            fea_list.append(f)
        
        hf['feature'].resize((n + 1, audio_num, 3, frames_num, mel_bins))
        hf['feature'][n] = fea_list
            
    hf.close()
        
    print('Write hdf5 file to {} using {:.3f} s'.format(
        feature_path, time.time() - extract_time))
    
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    # Calculate feature for all audio files
    parser_logmel = subparsers.add_parser('calculate_feature_for_all_audio_files')    
    parser_logmel.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')    
    parser_logmel.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')        
    parser_logmel.add_argument('--mini_data', action='store_true', default=False, help='Set True for debugging on a small part of data.')
        
    
    # Parse arguments
    args = parser.parse_args()
    calculate_feature_for_all_audio_files(args)

