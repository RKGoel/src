import os
import librosa
import scipy
from stft import *
import pickle

def load_data(upsampled_basedir, downsampled_basedir, data_section):
    upsampled_waves = []
    downsampled_waves = []
    upsampled_dir = os.path.join(upsampled_basedir, data_section)
    downsampled_dir = os.path.join(downsampled_basedir, data_section)
    # for dir in os.listdir(upsampled_dir):
    #     us_dir = os.path.join(upsampled_dir, dir)
    for filename in os.listdir(upsampled_dir):
        filedir = os.path.join(upsampled_dir, filename)
        # waveform, bitrate = librosa.load(filedir, sr=None, mono=True)
        bitrate, waveform = scipy.io.wavfile.read(filedir)
        upsampled_waves.append(waveform)

    # for dir in os.listdir(downsampled_dir):
    #     ds_dir = os.path.join(downsampled_dir, dir)
    #     print(ds_dir)
    for filename in os.listdir(downsampled_dir):
        filedir = os.path.join(downsampled_dir, filename)
        # waveform, bitrate = librosa.load(filedir, sr=None, mono=True)
        bitrate, waveform = scipy.io.wavfile.read(filedir)
        downsampled_waves.append(waveform)
    return np.array(upsampled_waves), np.array(downsampled_waves)

def feature_extract(data, fft_size, fs, overlap_fac=0.5, num_frames_to_input=1):
    magnitude = []
    phase = []
    for waveform in data:
        m, p = stft(waveform, fft_size, fs, overlap_fac)

        ## Append previous and next frames to every frame ##
        num_context = int((num_frames_to_input - 1) / 2)
        empty_frames = np.zeros((num_context, m.shape[1]), dtype=np.float32)
        modified_mag = np.zeros((m.shape[0], num_frames_to_input, m.shape[1]))

        m = np.concatenate((empty_frames, m))
        m = np.concatenate((m, empty_frames))

        # m.shape[0] would now become m.shape[0]+(2*num_context)
        # first non-zero entry in m is at num_context.
        for i in range(num_context, m.shape[0]-(2*num_context)):
            modified_mag[i] = m[i-num_context:i+num_context+1, :]

        if len(magnitude) == 0:
            magnitude = modified_mag
            phase = p
        else:
            magnitude = np.concatenate((magnitude, modified_mag), axis=0)
            phase = np.concatenate((phase, p), axis=0)
        # print(magnitude.shape)
    return np.array(magnitude), np.array(phase)

def feature_extract_wb(data, fft_size, fs, overlap_fac=0.5):
    magnitude = []
    phase = []
    for waveform in data:
        m, p = stft(waveform, fft_size, fs, overlap_fac)

        if len(magnitude) == 0:
            magnitude = m
            phase = p
        else:
            magnitude = np.concatenate((magnitude, m), axis=0)
            phase = np.concatenate((phase, p), axis=0)
        # print(magnitude.shape)
    return np.array(magnitude), np.array(phase)

def split_data(data, valid_frac, test_frac):
    train_frac = 1 - valid_frac - test_frac
    data_len = len(data)
    train_data = data[0:int(train_frac*data_len)]
    valid_data = data[int(train_frac*data_len):int((1-test_frac)*data_len)]
    test_data = data[int((1-test_frac)*data_len):]
    return train_data, valid_data, test_data

def normalize(data, band_type):
    normal_data = np.copy(data)
    mean = np.mean(normal_data, axis=0) # scale every feature based on mean on that feature values i.e. axis=0
    std_var = np.std(normal_data, axis=0)
    with open(band_type + 'mean.data', 'wb') as f:
        pickle.dump(mean, f)
    with open(band_type + 'std_var.data', 'wb') as f:
        pickle.dump(std_var, f)
    for i in range(len(normal_data)):
        normal_data[i] = (normal_data[i] - mean)/std_var
    return normal_data


def normalize_test(data, band_type="nb"):
    with open(band_type + 'mean.data', 'rb') as f:
        mean = pickle.load(f)
    with open(band_type + 'std_var.data', 'rb') as f:
        variance = pickle.load(f)
    return (data - mean) / variance

