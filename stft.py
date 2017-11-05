import numpy as np

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
def stft(data, fft_size, fs, overlap_fac=0.5):
    print "Data:", data
    print ""

    hop_size = np.int32(np.floor(fft_size * (1 - overlap_fac)))
    pad_end_size = fft_size  # the last segment can overlap the end of the data array by no more than one window size
    total_segments = np.int32(np.ceil(len(data) / np.float32(hop_size)))
    t_max = len(data) / np.float32(fs)

    window = np.hamming(fft_size)  # our half cosine window
    # inner_pad = np.zeros(fft_size)  # the zeros which will be used to double each segment size

    proc = np.concatenate((data, np.zeros(pad_end_size)))  # the data to process
    magnitude = np.empty((total_segments, fft_size), dtype=np.float32)  # space to hold the magnitude
    phase = np.empty((total_segments, fft_size), dtype=np.float32)  # space to hold the phase

    for i in xrange(total_segments):  # for each segment
        current_hop = hop_size * i  # figure out the current segment offset
        segment = proc[current_hop:current_hop + fft_size]  # get the current segment
        windowed = segment * window  # multiply by the half cosine function
        #padded = np.append(windowed, inner_pad)  # add 0s to double the length of the data
        #spectrum = np.fft.fft(windowed) / fft_size  # take the Fourier Transform and scale by the number of samples
        spectrum = np.fft.fft(windowed)  # take the Fourier Transform
        autopower = np.abs(spectrum * np.conj(spectrum))  # find the autopower spectrum
        magnitude[i, :] = autopower[:fft_size]  # append to the results array
        phase[i, :] = np.angle(spectrum)[:fft_size]

    # magnitude = 20 * np.log10(magnitude)  # scale to db
    magnitude = 2 * np.log(magnitude)  # X^M(k) = ln|X(k)|^2
    return magnitude, phase