from __future__ import print_function
import pickle
import json
import os
import math
from scipy.io import wavfile
from new_model import *
from stft import *
from wave_reconstruct import *
from myplotlib import *
from preprocess import *

## Load the data settings ##
settings_file = '../settings/data_settings.json'
settings = json.load(open(settings_file))
WB_FFT_SIZE = settings['wb_fft_size']
NB_FFT_SIZE = settings['nb_fft_size']
DATA_FREQ = settings['data_freq']
NYQUIST_FREQ = settings['nyquist_freq']
OVERLAP_FAC = settings['overlap_factor']
NUM_FRAMES_TO_INPUT = settings['num_frames_to_input']

## Load normalized features ##
with open('nb_train_mag.data', 'rb') as f:
    nb_train_magnitude = pickle.load(f)
with open('wb_train_mag.data', 'rb') as f:
    wb_train_magnitude = pickle.load(f)
# with open('nb_valid_mag.data', 'rb') as f:
#     nb_valid_magnitude = pickle.load(f)
# with open('wb_valid_mag.data', 'rb') as f:
#     wb_valid_magnitude = pickle.load(f)

## Split frequencies ##
print("Splitting frequencies...")
input_train_data = nb_train_magnitude[:,:, 0:int(NB_FFT_SIZE/2)]
output_train_data = wb_train_magnitude[:, int(WB_FFT_SIZE/4):int(WB_FFT_SIZE/2)]
# input_valid_data = nb_valid_magnitude[:, 0:int(NB_FFT_SIZE/2)]
# output_valid_data = wb_valid_magnitude[:, int(WB_FFT_SIZE/4):int(WB_FFT_SIZE/2)]
print("Frequencies splitted.")

## Debug ##
print("Shapes of training in & out data:")
print(input_train_data.shape)
print(output_train_data.shape)
# print("Shapes of validation in & out data:")
# print(input_valid_data.shape)
# print(output_valid_data.shape)

## Train & test neural network ##
print("Starting neural network training & optimization...")
test_inbasedir = settings['ds_dir_name_base']
test_outbasedir = settings['input_dir_name_base']
test_infile = os.path.join(test_inbasedir, "p225/p225_001.wav") # take first wave from training data
test_outfile = os.path.join(test_outbasedir, "p225/p225_001.wav") # take first wave from training data

test_bitrate_in, test_wave_in = wavfile.read(test_infile)
test_bitrate_out, test_wave_out = wavfile.read(test_outfile)
print(test_wave_in.shape[0])
print(test_wave_out.shape[0])

m, test_wave_in_phase = stft(test_wave_in, int(NB_FFT_SIZE), NYQUIST_FREQ, OVERLAP_FAC)

## Append previous and next frames to every frame ##
num_context = int((NUM_FRAMES_TO_INPUT - 1) / 2)
empty_frames = np.zeros((num_context, m.shape[1]), dtype=np.float32)
test_wave_in_mag = np.zeros((m.shape[0], NUM_FRAMES_TO_INPUT, m.shape[1]))

m = np.concatenate((empty_frames, m))
m = np.concatenate((m, empty_frames))

# m.shape[0] would now become m.shape[0]+(2*num_context)
# first non-zero entry in m is at num_context.
for i in range(num_context, m.shape[0]-(2*num_context)):
    test_wave_in_mag[i] = m[i-num_context:i+num_context+1, :]

# test_wave_in_mag, test_wave_in_phase = feature_extract(test_wave_in, int(NB_FFT_SIZE), NYQUIST_FREQ, OVERLAP_FAC, NUM_FRAMES_TO_INPUT)
# print(test_wave_in_mag.shape)
plot_wave(test_wave_in)
# plot_wave(test_wave_out)
# print(test_wave_in)

# reconstruct(test_wave_in_mag, test_wave_in_phase, int(NB_FFT_SIZE), int(NYQUIST_FREQ/4), test_wave_in_mag.shape[0], OVERLAP_FAC)

## TODO normalize test_mag

## test_wave_out is dummy right now ##
print(input_train_data.shape)
print(output_train_data.shape)
preds = run_model(input_train_data, output_train_data, test_wave_in_mag[:,:, 0:int(math.ceil(NB_FFT_SIZE/2))], test_wave_out)
print(preds.shape)
#
# ## Reconstruct the test wave ##
modified_Z_M = np.array(test_wave_in_mag[:,math.floor(NUM_FRAMES_TO_INPUT/2) , 0:int(math.ceil(NB_FFT_SIZE/2))]) + np.float64(2*math.log(2))
## TODO Un-normalize preds before concatenating
calc_mag = np.concatenate((modified_Z_M, preds), axis=1)
print(calc_mag.shape)
test_phase = test_wave_in_phase[:, 0:int(math.ceil(NB_FFT_SIZE/2))]
calc_phase = np.concatenate((test_phase, -np.flip(test_phase, 1)), axis=1)
# print(calc_phase.shape)
# plot_wave(test_wave_out)

reconstruct(calc_mag, calc_phase, int(WB_FFT_SIZE/2), int(DATA_FREQ/2), int(calc_mag.shape[0]), OVERLAP_FAC)