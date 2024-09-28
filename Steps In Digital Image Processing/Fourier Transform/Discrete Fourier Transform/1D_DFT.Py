import numpy as np
import matplotlib.pyplot as plt

# Sample signal
Fs = 1000  # Sampling frequency
T = 1 / Fs  # Sampling interval
t = np.arange(0, 1, T)  # Time vector
signal = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)

# Perform DFT
N = len(signal)
dft_values = np.fft.fft(signal)
frequencies = np.fft.fftfreq(N, T)

# Create subplots
plt.figure(figsize=(12, 6))

# Plot original signal in the spatial domain
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title("Original Spatial Domain Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()

# Plot frequency domain representation
plt.subplot(2, 1, 2)
plt.plot(frequencies[:N // 2], np.abs(dft_values)[:N // 2] * 2 / N)
plt.title("1D DFT Frequency Domain Representation")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()

plt.tight_layout()
plt.show()
