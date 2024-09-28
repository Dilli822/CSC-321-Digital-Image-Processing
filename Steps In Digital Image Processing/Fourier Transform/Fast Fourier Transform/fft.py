import numpy as np
import matplotlib.pyplot as plt

# Sample signal
Fs = 1000  # Sampling frequency
T = 1 / Fs  # Sampling interval
t = np.arange(0, 1, T)  # Time vector
f1 = 50  # Frequency of the first signal
f2 = 120  # Frequency of the second signal

# Create a sample signal
signal = 0.7 * np.sin(2 * np.pi * f1 * t) + 0.3 * np.sin(2 * np.pi * f2 * t)

# Perform FFT
N = len(signal)
fft_values = np.fft.fft(signal)
fft_freqs = np.fft.fftfreq(N, T)

# Plotting
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title("Time Domain Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(fft_freqs[:N // 2], np.abs(fft_values)[:N // 2] * 2 / N)
plt.title("Frequency Domain Signal (FFT)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")

plt.tight_layout()
plt.show()
