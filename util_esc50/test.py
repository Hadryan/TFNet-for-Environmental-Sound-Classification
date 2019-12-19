# visualize the feature maps

import torch
import numpy as np
from net import Cnns,Cnns2
from feature import *
from matplotlib import pyplot as plt
from net_vis import Cnns_deconv    
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
def add_noise(audio, percent=0.2):
    random_values = np.random.randn(len(audio))
    print(np.mean(random_values))
    print(np.abs(np.mean(audio)))
    out = audio + percent * random_values
    return out
def fre_noise(feature, percent=0.2):
    out = feature
    for i in range(13):
        random_values = np.random.randn(feature.shape[1])
        out[50+i, :] = out[50+i, :] + percent*random_values
    return out
def time_noise(feature, percent=0.2):
    out = feature
    for i in range(5):
        random_values = np.random.randn(feature.shape[0])
        out[:, 25+i] = out[:, 25+i] + percent*random_values
    return out

if __name__ == '__main__':
    Model = eval('Cnns')
    model = Model(50, activation='logsoftmax')
    checkpoint_path = '3400_iterations.pth'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    params=model.state_dict() 

    audio_path = '1-85362-A-0.wav'

    (audio, _) = read_audio(audio_path=audio_path, target_fs=44100)
    audio = pad_truncate_sequence(audio, 44100*5)
#     audio = add_noise(audio, percent=0.2)

    feature_extractor = LogMelExtractor(
        sample_rate=44100, 
        window_size=1764, 
        hop_size=882, 
        mel_bins=40, 
        fmin=50, 
        fmax=11025)
    feature = feature_extractor.transform(audio)
    feature = feature[0 : 250]
    feature = time_noise(feature, percent=3)
    x = np.transpose(feature, (1, 0))
    plt.imshow(x, cmap = plt.cm.jet)
    plt.axis('off')
    fig = plt.gcf()
    height, width = x.shape
    fig.set_size_inches(width/5./8.,height/5./8.)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.savefig("ori_conv_noise2.png", dpi=500, pad_inches = 0)
    plt.show()
    feature = torch.from_numpy(feature[None, :, :])
#     x1 = model.show(feature)
    x1, x2, x3, x4, out1, out2, out3, out4, out5, out6, out7= model.show(feature)
    
    feature = torch.squeeze(out4)
    x1 = np.transpose(feature.detach().numpy(), (1, 0))
    plt.imshow(x1, cmap = plt.cm.jet)
    plt.axis('off')
    fig = plt.gcf()
    height, width = x1.shape
    fig.set_size_inches(width/5./8.,height/5./8.)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.savefig("conv_noise2.png", dpi=500, pad_inches = 0)
    plt.show()
    
    (audio, _) = read_audio(audio_path=audio_path, target_fs=44100)
    audio = pad_truncate_sequence(audio, 44100*5)
#     audio = add_noise(audio, percent=0.2)
#     audio = data_pre(audio=audio, audio_length=5, fs=44100, audio_skip=0.1)
    feature_extractor = LogMelExtractor(
        sample_rate=44100, 
        window_size=1764, 
        hop_size=882, 
        mel_bins=40, 
        fmin=50, 
        fmax=11025)
    feature = feature_extractor.transform(audio)
    feature = feature[0 : 250]
    x = np.transpose(feature, (1, 0))
    plt.imshow(x, cmap = plt.cm.jet)
    plt.axis('off')
    fig = plt.gcf()
    height, width = x.shape
    fig.set_size_inches(width/5./8.,height/5./8.)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.savefig("ori_conv.png", dpi=500, pad_inches = 0)
    plt.show()
    feature = torch.from_numpy(feature[None, :, :])
#     x1 = model.show(feature)
    x1, x2, x3, x4, out1, out2, out3, out4, out5, out6, out7= model.show(feature)
    
    feature = torch.squeeze(out4)
    x1 = np.transpose(feature.detach().numpy(), (1, 0))
    plt.imshow(x1, cmap = plt.cm.jet)
    plt.axis('off')
    fig = plt.gcf()
    height, width = x1.shape
    fig.set_size_inches(width/5./8.,height/5./8.)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.savefig("conv.png", dpi=500, pad_inches = 0)
    plt.show()
    
    (audio, _) = read_audio(audio_path=audio_path, target_fs=44100)
    audio = pad_truncate_sequence(audio, 44100*5)
#     audio = add_noise(audio, percent=0.2)
#     audio = data_pre(audio=audio, audio_length=5, fs=44100, audio_skip=0.1)
    feature_extractor = LogMelExtractor(
        sample_rate=44100, 
        window_size=1764, 
        hop_size=882, 
        mel_bins=40, 
        fmin=50, 
        fmax=11025)
    feature = feature_extractor.transform(audio)
    feature = feature[0 : 250]
    feature = fre_noise(feature, percent=3)
    x = np.transpose(feature, (1, 0))
    plt.imshow(x, cmap = plt.cm.jet)
    plt.axis('off')
    fig = plt.gcf()
    height, width = x.shape
    fig.set_size_inches(width/5./8.,height/5./8.)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.savefig("ori_conv_noise1.png", dpi=500, pad_inches = 0)
    plt.show()
    feature = torch.from_numpy(feature[None, :, :])
#     x1 = model.show(feature)
    x1, x2, x3, x4, out1, out2, out3, out4, out5, out6, out7= model.show(feature)
    
    feature = torch.squeeze(out4)
    x1 = np.transpose(feature.detach().numpy(), (1, 0))
    plt.imshow(x1, cmap = plt.cm.jet)
    plt.axis('off')
    fig = plt.gcf()
    height, width = x1.shape
    fig.set_size_inches(width/5./8.,height/5./8.)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.savefig("conv_noise1.png", dpi=500, pad_inches = 0)
    plt.show()
    
    (audio, _) = read_audio(audio_path=audio_path, target_fs=44100)
    audio = pad_truncate_sequence(audio, 44100*5)
    audio = add_noise(audio, percent=0.08)
#     audio = data_pre(audio=audio, audio_length=5, fs=44100, audio_skip=0.1)
    feature_extractor = LogMelExtractor(
        sample_rate=44100, 
        window_size=1764, 
        hop_size=882, 
        mel_bins=40, 
        fmin=50, 
        fmax=11025)
    feature = feature_extractor.transform(audio)
    feature = feature[0 : 250]
    x = np.transpose(feature, (1, 0))
    plt.imshow(x, cmap = plt.cm.jet)
    plt.axis('off')
    fig = plt.gcf()
    height, width = x.shape
    fig.set_size_inches(width/5./8.,height/5./8.)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
#     plt.savefig("ori_conv_noise3.png", dpi=500, pad_inches = 0)
    plt.show()
    feature = torch.from_numpy(feature[None, :, :])
#     x1 = model.show(feature)
    x1, x2, x3, x4, out1, out2, out3, out4, out5, out6, out7= model.show(feature)
    
    feature = torch.squeeze(out4)
    x1 = np.transpose(feature.detach().numpy(), (1, 0))
    plt.imshow(x1, cmap = plt.cm.jet)
    plt.axis('off')
    fig = plt.gcf()
    height, width = x1.shape
    fig.set_size_inches(width/5./8.,height/5./8.)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.savefig("conv_noise3.png", dpi=500, pad_inches = 0)
    plt.show()

    
