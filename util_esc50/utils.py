import torch
import os
import sys
import numpy as np
import soundfile
import librosa
import h5py
import math
import pandas as pd
from sklearn import metrics
import logging
import matplotlib.pyplot as plt
import config

def mixup_data(x, y, alpha=0.2):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        while True:
            lam = np.random.beta(alpha, alpha)
            if lam > 0.65 or lam < 0.35 :
                break
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(class_criterion, pred, y_a, y_b, lam):
    return lam * class_criterion(pred, y_a) + (1 - lam) * class_criterion(pred, y_b)
def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

def get_filename(path):
    path = os.path.realpath(path)
    name_ext = path.split('/')[-1]
    name = os.path.splitext(name_ext)[0]
    return name


def create_logging(log_dir, filemode):
    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '{:04d}.log'.format(i1))):
        i1 += 1
        
    log_path = os.path.join(log_dir, '{:04d}.log'.format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return logging
    
def data_pre(audio, audio_length, fs, audio_skip):
    stride = int(audio_skip * fs /2)
    loop =  int((audio_length * fs) // stride - 1)
    area = 0
    maxamp = 0.
    i = 0
    out = audio
    while i < loop:
        win_data = out[i*stride: (i+2)*stride]
        maxamp = np.max(np.abs(win_data))
        if maxamp < 0.005:
            loop = loop - 2
            out[i*stride: (loop+1)*stride] = out[(i+2)*stride: (loop+3)*stride]
        else:
            i = i + 1
    length = (audio_length * fs) // stride - loop - 1
    if length == 0:
        return out
    else:
        out[(loop + 1) * stride:(audio_length * fs // stride) * stride] = out[0:length * stride]
        if length < (audio_length * fs//stride)/2:
            out[(loop+1)*stride:(audio_length * fs//stride)*stride] = out[0:length*stride]
            return out
        else:
            out[(loop + 1) * stride:(loop + 1)*2  * stride] = out[0:(loop + 1) * stride]
            return data_pre(out, audio_length, fs, audio_skip)

def read_audio(audio_path, target_fs=None):
    (audio, fs) = soundfile.read(audio_path)
#     audio = data_pre(audio=audio, audio_length=5, fs=fs, audio_skip=0.1)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
        
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
        
    return audio, fs
def read_audio_1D(audio_path, target_fs=None):
    (audio, fs) = soundfile.read(audio_path)
    audio = data_pre(audio=audio, audio_length=5, fs=fs, audio_skip=0.1)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
        
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
        
    return audio, fs


def read_left_audio(audio_path, target_fs=None):
    (audio, fs) = soundfile.read(audio_path)

    if audio.ndim > 1:
        audio = audio[0]

    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs

    return audio, fs

def read_side_audio(audio_path, target_fs=None):
    (audio, fs) = soundfile.read(audio_path)

    if audio.ndim > 1:
        audio = audio[0]-audio[1]

    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs

    return audio, fs

def read_right_audio(audio_path, target_fs=None):
    (audio, fs) = soundfile.read(audio_path)

    if audio.ndim > 1:
        audio = audio[1]

    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs

    return audio, fs
    
def pad_truncate_sequence(x, max_len):
# Data length Regularization
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0 : max_len]
    
   
    
def scale(x, mean, std):
    return (x - mean) / std
    
    
def inverse_scale(x, mean, std):
    return x * std + mean
    
        
        
def read_metadata(metadata_path):
    '''Read metadata from a csv file. 
    
    Returns:
      meta_dict: dict of meta data, e.g.:
         {'filename': np.array(['1-100032-A-0.wav', '1-100038-A-14.wav', ...]),
          'fold': np.array([1, 1, ...]),
           'target': np.array([0, 14, ...]),
           'category': np.array(['dog', 'chirping_birds', ...]),
           'esc10': np.array(['True', 'False', ...]),
           'src_file': np.array(['100032', '100038', ...]),
           'take': np.array(['A', 'A', ...])
         }
    '''
    
    df = pd.read_csv(metadata_path, sep=',')
    meta_dict = {}
    meta_dict['filename'] = np.array(
        [name for name in df['filename'].tolist()])
    
    if 'fold' in df.keys():
        meta_dict['fold'] = np.array(df['fold'])       
    if 'target' in df.keys():
        meta_dict['target'] = np.array(df['target'])
    if 'category' in df.keys():
        meta_dict['category'] = np.array(df['category'])
    if 'esc10' in df.keys():
        meta_dict['esc10'] = np.array(df['esc10'])
    if 'src_file' in df.keys():
        meta_dict['src_file'] = np.array(df['src_file'])
    if 'take' in df.keys():
        meta_dict['take'] = np.array(df['take'])
     
    return meta_dict
    
    
def sparse_to_categorical(x, n_out):
    x = x.astype(int)
    shape = x.shape
    x = x.flatten()
    N = len(x)
    x_categ = np.zeros((N,n_out))
    x_categ[np.arange(N), x] = 1
    return x_categ.reshape((shape)+(n_out,))
    
    
