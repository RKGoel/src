# Audio Upsampler using Deep Neural Network

## Feature Extraction Process

The downsampled audio files are read using scipy library, and based on a specific window size (fft_size), we slice them into number of overlapping windows with overlap factor of 0.5 (50%). On each window we perform short-time Fourier Transform (STFT) and feed it into the network after normalizing.
Network predicts the high frequency windows from low frequency windows and then we append the two windows respectively to generate upsampled wave. We perform wave reconstruction by passing appended windows to inverse fourier tranform with magnitude from network and estimation of phase.
Refer [this](http://ieeexplore.ieee.org/document/7178801/) paper for more information.

## File Structure

There are two json files which have settings for data (how data will be processed) and for deep neural network model (it's configuration, learning rate, iterations etc.). These two settings file should be in the folder named 'settings'.
Other python files should be in the 'src' folder within the same directory as 'settings'. Also, we will have a model log folder to save trained model files. Currently they are in 'tmp' folder within base direcotry. Now we have directory structure as:

```
basedir/settings
basedir/src
basedir/tmp
```

## Files Description

data_settings.json
* input_dir_name_base: base directory for input
* test_fraction, validation_fraction: self explanatory
* data_freq: original frequency of audio files
* wb_fft_size: window size for wideband (upsampled) audio
* nb_fft_size: window size for narrowband (downsampled) audio
* nyquist_freq: sampling frequency to read data (twice of original frequency)
* overlap_factor: overlap fraction for windows
* downsample_factor: by what factor to downsample
* num_frames_to_input: total number of frames (window) to input into network (current window plus previous and next windows)
* ds_dir_name_base: base directory for downsampled audio files

filter_downsample_save.py: loads the original upsampled audio files, filter and downsample each file and save them into downsampled base directory using specified downsampling rate.

save_features.py: loads the files, extract the features, normalize and save the features to files 'nb_train_mag.data' and 'wb_train_mag.data'

load_n_run_model.py: loads the extracted features from files, and run the model to train based on configurations specified in model_settings.json.

test_model.py: loads the trained model and tests on specific audio file and reconstructs it.

Other python files are almost self explanatory and are used by above three files internally.

## Running the scripts

* Set the file structure as specified
* Change the directory paths in settings files accordingly
* Run filter_downsample_save.py to save downsampled files
* Run save_features.py that extracts and saves features
* Now we can run our model several times by simply running load_n_run_model.py that loads extracted features and runs the model