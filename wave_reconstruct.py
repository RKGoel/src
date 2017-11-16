from scipy.io import wavfile
import matplotlib.pylab as plt
import numpy as np

def flip(nb_phase):
    wb_phase = np.empty(nb_phase.shape, dtype=np.float32)
    for i in range(nb_phase.shape[0]):
        wb_phase[i, :] = list(reversed(nb_phase[i, :]))
    return wb_phase

"""
Reconstruct audio signal based on magnitude array and phase array for overlapped segments

"""
def reconstruct(magnitude, phase, fft_size, sampling_freq, num_segments, overlap_fac=0.5):
    hop_size = np.int32(np.floor(fft_size * (1 - overlap_fac)))
    data_len = np.int32(np.ceil(num_segments*hop_size))
    window = np.hamming(fft_size)  # our half cosine window
    # num_segments = np.int32(np.ceil(data_len / np.float32(hop_size)))
    # print "Total Segments:", num_segments
    # print "Magnitude shape:", magnitude.shape
    # print "Phase shape:", phase.shape
    wave = np.empty((num_segments, fft_size), dtype=np.float32)
    rec_frame = np.zeros(data_len+fft_size)
    print (rec_frame.shape)

    for i in range(num_segments):
        complex = (magnitude[i]/2) + (1j*phase[i]) # calculate complex signal value for each segment
        #print np.exp(complex).shape
        #complex *= fft_size
        wave[i, :] = np.real(np.fft.ifft(np.exp(complex)))
        # wave[i] *= window
        current_hop = hop_size * i
        rec_frame[current_hop:current_hop+fft_size] += wave[i]
    print ("Wave shape after IIFT:", wave.shape)
    print ("Rec wave avg:", np.average(np.array(rec_frame[:data_len])))
    # print(rec_frame[:data_len])
    plt.plot(np.array(rec_frame[:data_len]))
    plt.show()
    wavfile.write('test.wav', sampling_freq, (rec_frame[:data_len]).astype(np.dtype('i2')))
