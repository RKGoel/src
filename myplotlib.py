import matplotlib.pylab as plt
import numpy as np

def plot_wave(wave):
    plt.plot(np.array(wave))
    plt.xlabel('Sample number')
    plt.ylabel('Amplitude')
    plt.show()

def plot_freq_heatmap(wave_magnitude):
    plt.imshow(np.array(wave_magnitude), origin='lower', cmap='jet', interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.xlabel("Sample number")
    plt.ylabel("Segment number")
    plt.show()