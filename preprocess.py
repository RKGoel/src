import os
import librosa
import scipy
from stft import *

def load_data(upsampled_basedir, downsampled_basedir):
    upsampled_waves = []
    downsampled_waves = []
    for dir in os.listdir(upsampled_basedir):
        full_dir = os.path.join(upsampled_basedir, dir)
        for filename in os.listdir(full_dir):
            filedir = os.path.join(full_dir, filename)
            waveform, bitrate = librosa.load(filedir, sr=None, mono=True)
            upsampled_waves.append(waveform)

    for dir in os.listdir(downsampled_basedir):
        ds_dir = os.path.join(downsampled_basedir, dir)
        for filename in os.listdir(full_dir):
            filedir = os.path.join(full_dir, filename)
            waveform, bitrate = librosa.load(filedir, sr=None, mono=True)
            downsampled_waves.append(waveform)
    return np.array(upsampled_waves), np.array(downsampled_waves)

def filter_downsample(waveform, original_sampling_freq, downsample_factor):
    ds_sampling_freq = original_sampling_freq / (2 * downsample_factor)
    ds_wave = scipy.signal.decimate(waveform, int(downsample_factor), ftype="fir")
    # print (waveform.shape)
    # print (ds_wave.shape)
    # print (ds_wave)
    # print(ds_dir+filename)
    # scipy.io.wavfile.write(os.path.join(ds_dir, filename), int(ds_sampling_freq), ds_wave[:])
    return ds_wave

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
    for i in range(len(normal_data)):
        normal_data[i] = normal_data[i] - np.mean(normal_data[i])/np.std(normal_data[i])
    return normal_data

