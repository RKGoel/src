from scipy.io import wavfile
from scipy.signal import decimate
import json
import os
import numpy as np
from myplotlib import plot_wave

## Load the data settings ##
settings_file = '../settings/data_settings.json'
settings = json.load(open(settings_file))

DOWNSAMPLE_FACTOR = settings['downsample_factor']
UP_BASEDIR = settings['input_dir_name_base']
DOWN_BASEDIR = settings['ds_dir_name_base']
DATA_SAMPLING_FREQ = settings['data_freq']

for dir in os.listdir(UP_BASEDIR):
    us_dir = os.path.join(UP_BASEDIR, dir)
    ds_dir = os.path.join(DOWN_BASEDIR, dir)
    if not os.path.exists(ds_dir):
        os.makedirs(ds_dir)
    for filename in os.listdir(us_dir):
        filedir = os.path.join(us_dir, filename)
        bitrate, waveform = wavfile.read(filedir)
        down_wave = decimate(waveform, int(DOWNSAMPLE_FACTOR), ftype="fir")
        # converted to astype(np.dtype('i2')) to make file VLC playable
        wavfile.write(os.path.join(ds_dir, filename), int(DATA_SAMPLING_FREQ/DOWNSAMPLE_FACTOR), down_wave[:].astype(np.dtype('i2')))
        down_bitrate, down_wave = wavfile.read(os.path.join(ds_dir, filename))
        # print(bitrate, down_bitrate)
        # print(waveform.shape, down_wave.shape)
        # plot_wave(waveform)
        # plot_wave(down_wave)
        # break
