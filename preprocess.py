import os
import librosa
import scipy
from stft import *

def load_data(upsampled_basedir, downsampled_basedir):
    upsampled_waves = []
    downsampled_waves = []
    for dir in os.listdir(upsampled_basedir):
        us_dir = os.path.join(upsampled_basedir, dir)
        for filename in os.listdir(us_dir):
            filedir = os.path.join(us_dir, filename)
            # waveform, bitrate = librosa.load(filedir, sr=None, mono=True)
            bitrate, waveform = scipy.io.wavfile.read(filedir)
            upsampled_waves.append(waveform)

    for dir in os.listdir(downsampled_basedir):
        ds_dir = os.path.join(downsampled_basedir, dir)
        print(ds_dir)
        for filename in os.listdir(ds_dir):
            filedir = os.path.join(ds_dir, filename)
            # waveform, bitrate = librosa.load(filedir, sr=None, mono=True)
            bitrate, waveform = scipy.io.wavfile.read(filedir)
            downsampled_waves.append(waveform)
    return np.array(upsampled_waves), np.array(downsampled_waves)

def feature_extract(data, fft_size, fs, overlap_fac=0.5):
    magnitude = []
    phase = []
    for waveform in data:
        m, p = stft(waveform, fft_size, fs, overlap_fac)
        if len(magnitude) == 0:
            magnitude = m
            phase = p
        else:
            magnitude = np.concatenate((magnitude, m), axis=0)
        # print(magnitude.shape)
    return np.array(magnitude), np.array(phase)

def split_data(data, valid_frac, test_frac):
    train_frac = 1 - valid_frac - test_frac
    data_len = len(data)
    train_data = data[0:int(train_frac*data_len)]
    valid_data = data[int(train_frac*data_len):int((1-test_frac)*data_len)]
    test_data = data[int((1-test_frac)*data_len):]
    return train_data, valid_data, test_data

def normalize(data):
    normal_data = np.copy(data)
    mean = np.mean(normal_data)
    std_var = np.std(normal_data)
    for i in range(len(normal_data)):
        normal_data[i] = (normal_data[i] - mean)/std_var
    return normal_data

