from __future__ import print_function
import math
from new_model import *
from wave_reconstruct import *
import pickle as pickle
#from preprocess import *

## Load the data settings ##
settings_file = 'data_settings.json'
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

run_model(input_train_data, output_train_data)
