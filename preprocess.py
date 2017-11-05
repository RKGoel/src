import os
import csv
import json
import numpy as np
import librosa
import matplotlib.pylab as plt
from scipy.io.wavfile import write,read
from stft import stft
from wave_reconstruct import reconstruct

settings_file = '../settings/data_settings.json'
settings = json.load(open(settings_file))

INPUT_DIR_BASE = settings['input_dir_name_base']
FFT_SIZE = settings['fft_size']
DATA_FREQ = settings['data_freq']
SAMPLING_FREQ = settings['nyquist_freq']
OVERLAP_FAC = settings['overlap_factor']

np.random.seed(0)

def flip(nb_phase):
    wb_phase = np.empty(nb_phase.shape, dtype=np.float32)
    for i in xrange(nb_phase.shape[0]):
        wb_phase[i, :] = list(reversed(nb_phase[i, :]))
    return wb_phase

for dir in os.listdir(INPUT_DIR_BASE):
    full_dir = os.path.join(INPUT_DIR_BASE, dir)
    for filename in os.listdir(full_dir):
        filedir = os.path.join(full_dir, filename)
        waveform, bitrate = librosa.load(filedir, sr=None, mono=True)
        #bitrate, waveform = read(filedir)
        channel = 1
        waveform.reshape((-1, channel))

        # Plot the wave
        plt.plot(np.array(waveform))
        plt.xlabel('Sample number')
        plt.ylabel('Amplitude')
        plt.show()

        wave_magnitude, wave_phase = stft(waveform, FFT_SIZE, SAMPLING_FREQ, OVERLAP_FAC)

        plt.imshow(np.array(wave_magnitude), origin='lower', cmap='jet', interpolation='nearest', aspect='auto')
        plt.colorbar()
        plt.xlabel("Sample number")
        plt.ylabel("Segment number")
        plt.show()

        nb_endpt = int(FFT_SIZE / 4)
        wb_endpt = int(FFT_SIZE / 2)
        nb_magnitude = wave_magnitude[:, 0:nb_endpt]
        wb_magnitude = wave_magnitude[:, nb_endpt + 1:wb_endpt + 1]
        nb_phase = wave_phase[:, 0:nb_endpt]
        wb_phase = wave_phase[:, nb_endpt + 1:wb_endpt + 1]

        #print -flip(nb_phase)[0]
        print nb_phase[0]
        print ""
        print wb_phase[0]

        normalized_nb = (nb_magnitude - np.mean(nb_magnitude))/np.std(nb_magnitude)
        normalized_wb = (wb_magnitude - np.mean(wb_magnitude))/np.std(wb_magnitude)

        #reconstruct(narrowband_magnitude, narrowband_phase, FFT_SIZE/4, DATA_FREQ/4, len(waveform)/4)
        #reconstruct(wave_magnitude, wave_phase, FFT_SIZE, DATA_FREQ, len(waveform))
        #reconstruct(wideband_magnitude, wideband_phase, FFT_SIZE/4, DATA_FREQ/4, len(waveform)/4)
        break

# low = "low_freq_rec.wav"
# high = "high_freq_rec.wav"
# low_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), low)
# high_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), high)
# low_bt, low_wave = read(low_file)
# high_bt, high_wave = read(high_file)
# print low_wave+high_wave
# print ""
# print low_wave
# print low_bt
# print high_wave
# print high_bt