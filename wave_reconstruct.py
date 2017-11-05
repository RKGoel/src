from scipy.io.wavfile import write,read
import matplotlib.pylab as plt
import numpy as np

"""
Reconstruct audio signal based on magnitude array and phase array for overlapped segments

"""
def reconstruct(magnitude, phase, fft_size, sampling_freq, data_len, overlap_fac=0.5):
    hop_size = np.int32(np.floor(fft_size * (1 - overlap_fac)))
    num_segments = np.int32(np.ceil(data_len / np.float32(hop_size)))
    #num_segments = len(magnitude)
    # print "Total Segments:", num_segments
    # print "Magnitude shape:", magnitude.shape
    # print "Phase shape:", phase.shape
    wave = np.empty((num_segments, fft_size), dtype=np.float32)
    rec_frame = np.zeros(data_len+fft_size)
    print rec_frame.shape

    for i in xrange(num_segments):
        complex = (magnitude[i]/2) + (1j*phase[i]) # calculate complex signal value for each segment
        #print np.exp(complex).shape
        wave[i, :] = np.real(np.fft.ifft(np.exp(complex)))
        current_hop = hop_size * i
        rec_frame[current_hop:current_hop+fft_size] += wave[i]
    print "Wave shape after IIFT:", wave.shape
    print np.array(rec_frame[:data_len])
    plt.plot(np.array(rec_frame[:data_len]))
    plt.show()
    write('test.wav', sampling_freq, rec_frame[:data_len])
