import numpy as np
import matplotlib.pyplot as plt

# Set a random seed for reproducibility
np.random.seed(0)

# Generate a random sampling rate between 500 and 2000 Hz
sampling_rate = np.random.randint(500, 5000)  # Random integer between 500 and 2000
T = 1.0 / sampling_rate  # Sampling interval
L = 1.0  # Length of the signal in seconds
N = int(L / T)  # Number of samples
x = np.linspace(0.0, L, N, endpoint=False)  # Time vector

# Create a signal with a frequency of 50 Hz
frequency = 50  # Frequency in Hz
signal = np.sin(2 * np.pi * frequency * x)  # Sine wave signal

# Compute the 1D Discrete Fourier Transform (DFT)
def dft_1d(signal):
    N = len(signal)
    dft_result = np.zeros(N, dtype=complex)
    for k in range(N):
        sum_val = 0
        for n in range(N):
            angle = -2j * np.pi * k * n / N
            sum_val += signal[n] * np.exp(angle)
        dft_result[k] = sum_val
    return dft_result

# Compute DFT
dft_result = dft_1d(signal)

# Compute magnitude spectrum
def magnitude_spectrum(frequency_domain):
    return np.abs(frequency_domain) / len(frequency_domain)  # Normalize

magnitude_spectrum_result = magnitude_spectrum(dft_result)

# Frequency axis
freqs = np.fft.fftfreq(N, T)  # Frequency bins

# Only take the positive half of the spectrum
positive_freqs = freqs[:N//2]
positive_magnitude_spectrum = magnitude_spectrum_result[:N//2]

# Plotting the results
plt.figure(figsize=(12, 6))

# Plot the original signal
plt.subplot(2, 1, 1)
plt.plot(x, signal)
plt.title(f'Original Signal (Sampling Rate = {sampling_rate} Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

# Plot the magnitude spectrum
plt.subplot(2, 1, 2)
plt.plot(positive_freqs, positive_magnitude_spectrum)
plt.title('Magnitude Spectrum (1D DFT)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.grid()

plt.tight_layout()
plt.show()

