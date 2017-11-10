import json
from preprocess import *
from model import *
from sush_model import start_DNN

## Load the data settings ##
settings_file = '../settings/data_settings.json'
settings = json.load(open(settings_file))

INPUT_DIR_BASE = settings['input_dir_name_base']
DOWNSAMPLE_DIR_BASE = settings['ds_dir_name_base']
WB_FFT_SIZE = settings['wb_fft_size']
NB_FFT_SIZE = settings['nb_fft_size']
DATA_FREQ = settings['data_freq']
SAMPLING_FREQ = settings['nyquist_freq']
OVERLAP_FAC = settings['overlap_factor']
DOWNSAMPLE_FAC = settings['downsample_factor']
VALID_FRAC = settings['validation_fraction']
TEST_FRAC = settings['test_fraction']

## Load the data using methods in preprocess.py ##
upsampled_waves, downsampled_waves = load_data(INPUT_DIR_BASE, DOWNSAMPLE_DIR_BASE)

## Split the data in training, validation & test set ###
wb_train_data, wb_valid_data, wb_test_data = split_data(upsampled_waves, VALID_FRAC, TEST_FRAC)
nb_train_data, nb_valid_data, nb_test_data = split_data(downsampled_waves, VALID_FRAC, TEST_FRAC)
print(wb_train_data.shape, wb_valid_data.shape, wb_test_data.shape)

## Extract features ##
wb_magnitude, wb_phase = feature_extract(wb_train_data, WB_FFT_SIZE, SAMPLING_FREQ, OVERLAP_FAC)
nb_magnitude, nb_phase = feature_extract(nb_train_data, NB_FFT_SIZE, SAMPLING_FREQ/DOWNSAMPLE_FAC, OVERLAP_FAC)
print(wb_magnitude.shape)
print(nb_magnitude.shape)

## Normalize features ##
nb_magnitude = normalize(nb_magnitude)
wb_magnitude = normalize(wb_magnitude)

## Split frequencies ##
input_data = nb_magnitude[:, 0:int(NB_FFT_SIZE/2)]
output_data = wb_magnitude[:, int(WB_FFT_SIZE/4):int(WB_FFT_SIZE/2)]
# train_fn(input_data, output_data)
start_DNN(input_data, output_data)