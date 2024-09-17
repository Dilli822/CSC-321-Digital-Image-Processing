import numpy as np
import matplotlib.pyplot as plt
import random

# Generate random parameters
sampling_rate = random.randint(500, 2000)  # Hz
duration = random.uniform(0.5, 2.0)  # seconds
frequency1 = random.randint(5, 20)  # Hz
frequency2 = random.randint(30, 50)  # Hz

# Generate a simple signal
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
signal = np.sin(2 * np.pi * frequency1 * t) + 0.5 * np.sin(2 * np.pi * frequency2 * t)

# Compute the spectrum
spectrum = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(t), 1/sampling_rate)

# Plot
plt.figure(figsize=(12, 8))

# Signal plot
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title(f'Original Signal (Sampling Rate: {sampling_rate} Hz, Duration: {duration:.2f} s)')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')

# Spectrum plot
plt.subplot(2, 1, 2)
plt.plot(frequencies[:len(frequencies)//2], np.abs(spectrum)[:len(frequencies)//2])
plt.title('Spectral Analysis')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

# Find the two highest peaks
magnitude_spectrum = np.abs(spectrum[:len(frequencies)//2])
peak_indices = np.argsort(magnitude_spectrum)[-2:]
peak_freqs = frequencies[peak_indices]

# Add dynamic explanatory text
plt.figtext(0.5, 0.01, 
            f"Spectral analysis breaks down a signal into its component frequencies.\n"
            f"Peaks in the spectrum show the main frequencies present in the signal.\n"
            f"Here, we see prominent peaks at approximately {peak_freqs[1]:.1f} Hz and {peak_freqs[0]:.1f} Hz,\n"
            f"corresponding to the main frequencies in our original signal.\n"
            f"The sampling rate of {sampling_rate} Hz allows us to detect frequencies up to {sampling_rate/2} Hz.",
            ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

plt.tight_layout()
plt.subplots_adjust(bottom=0.25)  # Make room for the text
plt.show()