import tensorflow as tf
import json
import os
import math
import scipy
import numpy as np
from scipy.io import wavfile


def segSNR(test_wave_in_mag, test_wave_in_mag_generated, L, N):
    # print(test_wave_in_mag.shape)
    # print(test_wave_in_mag_generated.shape)
    # print("SNR start")
    val = 0
    for l in range(L):
        num = 0
        den = 0
        for n in range(N):
            num += np.square(test_wave_in_mag[l, n])
            den += np.square(np.subtract(test_wave_in_mag[l, n], test_wave_in_mag_generated[l, n]))
        val += 10 * np.log10(num / den)

    return val / L


def LSD(test_wave_in_mag, test_wave_in_mag_generated, L, N):
    # print("LSD start")
    val = 0
    for l in range(L):
        value = 0
        for n in range((N // 2) + 1):
            value += np.square(np.subtract(test_wave_in_mag[l, n], test_wave_in_mag_generated[l, n]))
        val += np.sqrt((1 / ((N / 2) + 1)) * value)

    return val / L


def stft(data, fft_size, fs, overlap_fac=0.5):
    """
        Perform Short-time Fourier Transform on given audio signal

        @Params:
        data: a numpy array containing the signal to be processed
        fft_size: a scalar, window/segment size (number of samples in a window)
        fs: a scalar, Sampling frequency of data (which is twice of signal frequency)
        overlap_fac: a float in range [0,1], Overlap factor for windows/segments

        @Returns:
        magnitude: a numpy 2D array having value of log-spectral magnitude
                   for ith segment at jth sample
        phase: a numpy 2D array having value of phase angle for ith
               segment at jth sample
     """

    # print ("Data:", data)
    # print ("")

    hop_size = np.int32(np.floor(fft_size * (1 - overlap_fac)))
    pad_end_size = fft_size  # the last segment can overlap the end of the data array by no more than one window size
    total_segments = np.int32(np.ceil(int(data.shape[0]) / np.float32(hop_size)))
    t_max = data.shape[0] / np.float32(fs)

    window = np.hamming(fft_size)  # our half cosine window
    # inner_pad = np.zeros(fft_size)  # the zeros which will be used to double each segment size

    proc = np.concatenate((data, np.zeros(pad_end_size)))  # the data to process
    magnitude = np.empty((total_segments, fft_size), dtype=np.float32)  # space to hold the magnitude
    phase = np.empty((total_segments, fft_size), dtype=np.float32)  # space to hold the phase

    for i in range(total_segments):  # for each segment
        current_hop = hop_size * i  # figure out the current segment offset
        segment = proc[current_hop:current_hop + fft_size]  # get the current segment
        windowed = segment * window  # multiply by the half cosine function
        # padded = np.append(windowed, inner_pad)  # add 0s to double the length of the data
        # spectrum = np.fft.fft(windowed) / fft_size  # take the Fourier Transform and scale by the number of samples
        spectrum = np.fft.fft(windowed)  # take the Fourier Transform
        autopower = np.absolute(spectrum)  # find the autopower spectrum
        magnitude[i, :] = autopower[:fft_size]  # append to the results array
        phase[i, :] = np.angle(spectrum)[:fft_size]

    # magnitude = 20 * np.log10(magnitude)  # scale to db
    magnitude = 2 * np.log(magnitude)  # X^M(k) = ln|X(k)|^2
    return magnitude, phase


def reconstruct(magnitude, phase, fft_size, num_segments, overlap_fac=0.5, name='test.wav'):
    """
    Reconstruct audio signal based on magnitude array and phase array for overlapped segments
    """
    hop_size = np.int32(np.floor(fft_size * (1 - overlap_fac)))
    data_len = np.int32(np.ceil(num_segments * hop_size))
    # num_segments = np.int32(np.ceil(data_len / np.float32(hop_size)))
    # num_segments = len(magnitude)
    # print "Total Segments:", num_segments
    # print "Magnitude shape:", magnitude.shape
    # print "Phase shape:", phase.shape
    wave = np.empty((num_segments, fft_size), dtype=np.float32)
    wave_2 = np.empty((num_segments, fft_size), dtype=np.float32)
    rec_frame = np.zeros(data_len + fft_size)
    # print(rec_frame.shape)

    for i in range(num_segments):
        complex = (magnitude[i] / 2) + (1j * phase[i])  # calculate complex signal value for each segment
        # print np.exp(complex).shape
        # complex *= fft_size
        wave[i, :] = np.real(np.fft.ifft(np.exp(complex)))
        wave_2[i, :] = np.real(np.fft.ifft(np.exp(complex * 2)))
        current_hop = hop_size * i
        rec_frame[current_hop:current_hop + fft_size] += wave[i]
    # print("Wave shape after IIFT:", wave.shape)
    # print("Rec wave avg:", np.average(np.array(rec_frame[:data_len])))
    # plt.plot(np.array(rec_frame[:data_len]))
    # plt.show()

    return rec_frame, data_len


def load_and_reconstruct():
    test_directory = 'D:/Sush_MS/Python/Exercise Files/DLProject/EnglishSpeechUpsampler-master/sush_new/test_sample/'

    model_settings_file = 'settings/model_settings.json'
    settings = json.load(open(model_settings_file))
    model_save_directory = settings['model_save_directory'] # D:/Sush_MS/Python/Exercise Files/DLProject/EnglishSpeechUpsampler-master/sush_new/model/
    model_name = settings['model_name'] #savedmodel

    data_setting_file = 'settings/data_settings.json'
    settings = json.load(open(data_setting_file))
    WB_FFT_SIZE = settings['wb_fft_size'] #512
    NB_FFT_SIZE = settings['nb_fft_size'] #256
    DATA_FREQ = settings['data_freq'] #48000
    NYQUIST_FREQ = settings['nyquist_freq'] #96000
    OVERLAP_FAC = settings['overlap_factor'] #0.5

    current_dirs_parent = os.path.dirname(os.path.dirname(test_directory))
    test_directory_output = os.path.join(current_dirs_parent, 'test_output')

    if not os.path.exists(test_directory_output):
        os.makedirs(test_directory_output)

    with tf.Session() as session:
        saver = tf.train.import_meta_graph(model_save_directory + model_name + ".meta")
        saver.restore(session, tf.train.latest_checkpoint(model_save_directory))
        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name('X:0')
        preds = graph.get_tensor_by_name('preds:0')
        total_tests = len(os.listdir(test_directory))
        total_snr = 0
        total_lsd = 0
        for file_name in os.listdir(test_directory):
            test_bitrate_in, test_wave_in = wavfile.read(os.path.join(test_directory, file_name))
            test_wave_in_mag, test_wave_in_phase = stft(test_wave_in, int(NB_FFT_SIZE), NYQUIST_FREQ,
                                                        OVERLAP_FAC)
            prediction = session.run(preds, feed_dict={X: test_wave_in_mag[:, 0:int(math.ceil(NB_FFT_SIZE / 2))]})
            new_Z_M = np.array(test_wave_in_mag[:, 0:int(math.ceil(NB_FFT_SIZE / 2))]) + np.float64(2 * math.log(2))
            calc_mag = np.concatenate((new_Z_M, prediction), axis=1)
            test_phase = test_wave_in_phase[:, 0:int(math.ceil(NB_FFT_SIZE / 2))]
            calc_phase = np.concatenate((test_phase, -np.flip(test_phase, 1)), axis=1)
            resconstructed_wave, data_len = reconstruct(calc_mag, calc_phase, int(WB_FFT_SIZE / 2),
                                                        int(calc_mag.shape[0]), OVERLAP_FAC)
            # file_name_recon = 'recon_' + file_name

            name = os.path.join(test_directory_output, file_name)
            scipy.io.wavfile.write(name, int(DATA_FREQ / 2), resconstructed_wave[:data_len].astype(np.dtype('i2')))

            wave_to_be_passed, _ = stft(resconstructed_wave, int(WB_FFT_SIZE / 2), int(DATA_FREQ / 2))
            snr = segSNR(test_wave_in_mag, np.array(wave_to_be_passed), test_wave_in_mag.shape[0],
                         test_wave_in_mag.shape[1])
            # print(snr)
            total_snr += snr
            lsd = LSD(test_wave_in_mag, np.array(wave_to_be_passed), test_wave_in_mag.shape[0],
                      test_wave_in_mag.shape[1])
            # print(lsd)
            total_lsd += lsd

    print("Total {} files up-sampled and reconstructed".format(total_tests))
    print("Final SNR:", total_snr / total_tests)
    print("Final LSD:", total_lsd / total_tests)


if __name__ == '__main__':
    load_and_reconstruct()
