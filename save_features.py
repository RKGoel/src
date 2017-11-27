from __future__ import print_function
import json
import pickle
from preprocess import *
from new_model import *

## Load the data settings ##
settings_file = '../settings/data_settings.json'
settings = json.load(open(settings_file))

INPUT_DIR_BASE = settings['input_dir_name_base']
DOWNSAMPLE_DIR_BASE = settings['ds_dir_name_base']
WB_FFT_SIZE = settings['wb_fft_size']
NB_FFT_SIZE = settings['nb_fft_size']
DATA_FREQ = settings['data_freq']
NYQUIST_FREQ = settings['nyquist_freq']
OVERLAP_FAC = settings['overlap_factor']
DOWNSAMPLE_FAC = settings['downsample_factor']
VALID_FRAC = settings['validation_fraction']
TEST_FRAC = settings['test_fraction']
NUM_FRAMES_TO_INPUT = settings['num_frames_to_input']

## Load the data using methods in preprocess.py ##
print("Loading audio files...")
# upsampled_waves, downsampled_waves = load_data(INPUT_DIR_BASE, DOWNSAMPLE_DIR_BASE)
# print(len(upsampled_waves), "audio files loaded.")
# # print(upsampled_waves[0].shape, downsampled_waves[0].shape)
#
# ## Split the data in training, validation & test set ###
# wb_train_data, wb_valid_data, wb_test_data = split_data(upsampled_waves, VALID_FRAC, TEST_FRAC)
# nb_train_data, nb_valid_data, nb_test_data = split_data(downsampled_waves, VALID_FRAC, TEST_FRAC)

wb_train_data, nb_train_data = load_data(INPUT_DIR_BASE, DOWNSAMPLE_DIR_BASE, "train")
wb_valid_data, nb_valid_data = load_data(INPUT_DIR_BASE, DOWNSAMPLE_DIR_BASE, "dev")
wb_test_data, nb_test_data = load_data(INPUT_DIR_BASE, DOWNSAMPLE_DIR_BASE, "test")
print("Data splitted into training, validation & testing as follows:")
print(wb_train_data.shape, wb_valid_data.shape, wb_test_data.shape)


print("Extracting features...")
## Extract training features ##
wb_train_magnitude, wb_train_phase = feature_extract_wb(wb_train_data, WB_FFT_SIZE, NYQUIST_FREQ, OVERLAP_FAC)
nb_train_magnitude, nb_train_phase = feature_extract(nb_train_data, NB_FFT_SIZE, NYQUIST_FREQ/DOWNSAMPLE_FAC, OVERLAP_FAC, NUM_FRAMES_TO_INPUT)
print(wb_train_magnitude.shape)
print(nb_train_magnitude.shape)
print("Training features extracted.")

# Extract validation features ##
# wb_valid_magnitude, wb_valid_phase = feature_extract(wb_valid_data, WB_FFT_SIZE, NYQUIST_FREQ, OVERLAP_FAC, NUM_FRAMES_TO_INPUT)
# nb_valid_magnitude, nb_valid_phase = feature_extract(nb_valid_data, NB_FFT_SIZE, NYQUIST_FREQ/DOWNSAMPLE_FAC, OVERLAP_FAC, NUM_FRAMES_TO_INPUT)
# print(wb_valid_magnitude.shape)
# print(nb_valid_magnitude.shape)
# print("Validation features extracted.")

## Normalize features ##
print("Normalizing features...")
nb_train_magnitude = normalize(nb_train_magnitude, "nb") # train mean and variance should also be used for testing
wb_train_magnitude = normalize(wb_train_magnitude, "wb")
# nb_valid_magnitude = normalize(nb_valid_magnitude)
# wb_valid_magnitude = normalize(wb_valid_magnitude)
print("Features normalized.")

## Save normalized features ##
with open('nb_train_mag.data', 'wb') as f:
    pickle.dump(nb_train_magnitude, f)
with open('wb_train_mag.data', 'wb') as f:
    pickle.dump(wb_train_magnitude, f)
# with open('nb_valid_mag.data', 'wb') as f:
#     pickle.dump(nb_valid_magnitude, f)
# with open('wb_valid_mag.data', 'wb') as f:
#     pickle.dump(wb_valid_magnitude, f)

# ## Split frequencies ##
# print("Splitting frequencies...")
# input_train_data = nb_train_magnitude[:, 0:int(NB_FFT_SIZE/2)]
# output_train_data = wb_train_magnitude[:, int(WB_FFT_SIZE/4):int(WB_FFT_SIZE/2)]
# input_valid_data = nb_valid_magnitude[:, 0:int(NB_FFT_SIZE/2)]
# output_valid_data = wb_valid_magnitude[:, int(WB_FFT_SIZE/4):int(WB_FFT_SIZE/2)]
# print("Frequencies splitted.")
#
# ## Debug ##
# print("Shapes of training in & out data:")
# print(input_train_data.shape)
# print(output_train_data.shape)
# print("Shapes of validation in & out data:")
# print(input_valid_data.shape)
# print(output_valid_data.shape)


