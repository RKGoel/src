import os
import csv
import json
import numpy as np
import librosa
import matplotlib.pylab as plt
from scipy.io.wavfile import write, read
from stft import stft
from wave_reconstruct import reconstruct

###
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
###

# settings_file = 'data_settings.json'
# settings = json.load(open(settings_file))
#
# INPUT_DIR_BASE = settings['input_dir_name_base']
# FFT_SIZE = settings['fft_size']
# DATA_FREQ = settings['data_freq']
# SAMPLING_FREQ = settings['nyquist_freq']
# OVERLAP_FAC = settings['overlap_factor']
#
# np.random.seed(0)
#
#
# def flip(nb_phase):
#     wb_phase = np.empty(nb_phase.shape, dtype=np.float32)
#     for i in range(nb_phase.shape[0]):
#         wb_phase[i, :] = list(reversed(nb_phase[i, :]))
#     return wb_phase
#
#
# def start_process():
#     print("start process")
#     for dir in os.listdir(INPUT_DIR_BASE):
#         full_dir = os.path.join(INPUT_DIR_BASE, dir)
#         for filename in os.listdir(full_dir):
#             filedir = os.path.join(full_dir, filename)
#             waveform, bitrate = librosa.load(filedir, sr=None, mono=True)
#             # bitrate, waveform = read(filedir)
#             channel = 1
#             waveform.reshape((-1, channel))
#
#             # Plot the wave
#             plt.plot(np.array(waveform))
#             plt.xlabel('Sample number')
#             plt.ylabel('Amplitude')
#             # plt.show()
#
#             wave_magnitude, wave_phase = stft(waveform, FFT_SIZE, SAMPLING_FREQ, OVERLAP_FAC)
#
#             plt.imshow(np.array(wave_magnitude), origin='lower', cmap='jet', interpolation='nearest', aspect='auto')
#             plt.colorbar()
#             plt.xlabel("Sample number")
#             plt.ylabel("Segment number")
#             # plt.show()
#
#             nb_endpt = int(FFT_SIZE / 4)
#             wb_endpt = int(FFT_SIZE / 2)
#             nb_magnitude = wave_magnitude[:, 0:nb_endpt]
#             wb_magnitude = wave_magnitude[:, nb_endpt + 1:wb_endpt + 1]
#             nb_phase = wave_phase[:, 0:nb_endpt]
#             wb_phase = wave_phase[:, nb_endpt + 1:wb_endpt + 1]
#
#             # print( -flip(nb_phase)[0]
#             # print(nb_phase[0])
#             # print("")
#             # print(wb_phase[0])
#
#             normalized_nb = (nb_magnitude - np.mean(nb_magnitude)) / np.std(nb_magnitude)
#             normalized_wb = (wb_magnitude - np.mean(wb_magnitude)) / np.std(wb_magnitude)
#
#             # print("shape of nb:{}, wb: {}".format(normalized_nb.shape, normalized_wb.shape))
#             start_DNN(normalized_nb, normalized_wb)
#
#             # reconstruct(narrowband_magnitude, narrowband_phase, FFT_SIZE/4, DATA_FREQ/4, len(waveform)/4)
#             # reconstruct(wave_magnitude, wave_phase, FFT_SIZE, DATA_FREQ, len(waveform))
#             # reconstruct(wideband_magnitude, wideband_phase, FFT_SIZE/4, DATA_FREQ/4, len(waveform)/4)
#             break
#
#             # low = "low_freq_rec.wav"
#             # high = "high_freq_rec.wav"
#             # low_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), low)
#             # high_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), high)
#             # low_bt, low_wave = read(low_file)
#             # high_bt, high_wave = read(high_file)
#             # print( low_wave+high_wave)
#             # print( "")
#             # print( low_wave)
#             # print( low_bt)
#             # print( high_wave)
#             # print( high_bt)


def start_DNN(normalized_nb, normalized_wb):
    print("Start DNN now")
    print(normalized_nb.shape, normalized_wb.shape)

    def init_weights(shape):
        return tf.Variable(tf.random_normal(shape, stddev=0.01))

    # this network is the same as the previous one except with an extra hidden layer + dropout
    def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden):
        X = tf.nn.dropout(X, p_keep_input)
        h = tf.nn.relu(tf.matmul(X, w_h))

        h = tf.nn.dropout(h, p_keep_hidden)
        h2 = tf.nn.relu(tf.matmul(h, w_h2))

        h2 = tf.nn.dropout(h2, p_keep_hidden)

        return tf.matmul(h2, w_o)

    # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    teX, teY = trX, trY = normalized_nb, normalized_wb

    X = tf.placeholder("float", [None, 256])
    # X = tf.placeholder("float", [None, 784])
    # Y = tf.placeholder("float", [None, 10])
    Y = tf.placeholder("float", [None, 128])

    # w_h = init_weights([784, 625])
    w_h = init_weights([256, 20])
    # w_h2 = init_weights([625, 625])
    w_h2 = init_weights([20, 20])
    # w_o = init_weights([625, 10])
    w_o = init_weights([20, 128])

    p_keep_input = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")
    py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    predict_op = tf.argmax(py_x, 1)

    # Launch the graph in a session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        with tf.device('/device:GPU:0'):
            # you need to initialize all variables
            sess.run(tf.global_variables_initializer())
            for i in range(10):
                for start, end in zip(range(0, len(trX), 128), range(128, len(trX) + 1, 128)):
                    sess.run(train_op,
                             feed_dict={X: trX[start:end], Y: trY[start:end], p_keep_input: 0.8, p_keep_hidden: 0.5})

                print(i, np.mean(np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict={X: teX, p_keep_input: 1.0,
                                                                                           p_keep_hidden: 1.0})))


