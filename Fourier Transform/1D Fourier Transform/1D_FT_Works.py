
import numpy as np
import matplotlib.pyplot as plt

simplest_explain = """

Here’s how the code 1D_Fourier transform works with the simplest example, step-by-step:

Create a Tune: The code starts by making a simple musical note (a sine wave) with a frequency of 50 Hz. This is like playing a single note on a musical instrument.

Listen to the Tune: To understand the tune better, we need to "listen" to it in a special way. This special listening is called the Fourier Transform. It helps us see which notes (frequencies) are in the tune and how loud each one is.

Show the Notes: The code then shows us two things:

The Tune: What the original note looks like when we play it over time.
The Notes Breakdown: How loud each note is, and what the frequency of each note is.

"""

# Set a random seed for reproducibility
np.random.seed(0)

# Generate a random sampling rate between 500 and 5000 Hz
sampling_rate = np.random.randint(500, 5000)  # Random integer between 500 and 5000
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

# Calculate highest, lowest, and average frequency
def frequency_analysis(freqs, magnitude_spectrum):
    # Find indices of non-zero magnitudes to avoid zero frequency
    non_zero_indices = np.where(magnitude_spectrum > 0)[0]
    highest_freq = freqs[non_zero_indices[-1]] if non_zero_indices.size > 0 else 0
    lowest_freq = freqs[non_zero_indices[0]] if non_zero_indices.size > 0 else 0
    average_freq = np.sum(freqs * magnitude_spectrum) / np.sum(magnitude_spectrum) if np.sum(magnitude_spectrum) != 0 else 0
    
    return highest_freq, lowest_freq, average_freq

highest_freq, lowest_freq, average_freq = frequency_analysis(positive_freqs, positive_magnitude_spectrum)

# Plotting the results
plt.figure(figsize=(12, 8))

# Plot the original signal
plt.subplot(3, 1, 1)
plt.plot(x, signal)
plt.title(f'Original Signal (Sampling Rate = {sampling_rate} Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

# Plot the magnitude spectrum
plt.subplot(3, 1, 2)
plt.plot(positive_freqs, positive_magnitude_spectrum, color='blue')
plt.title('Magnitude Spectrum (1D DFT)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.grid()
plt.xlim(0, 150)  # Limit the x-axis to a reasonable range

# Explanation and frequency analysis plot
plt.subplot(3, 1, 3)
plt.text(0.1, 0.8, '1. Generate a signal with a frequency of 50 Hz.', fontsize=12)
plt.text(0.1, 0.6, '2. Compute the Discrete Fourier Transform (DFT) to analyze the signal.', fontsize=12)
plt.text(0.1, 0.4, '3. Calculate the magnitude of the DFT result to get the frequency spectrum.', fontsize=12)
plt.text(0.1, 0.0, f'4. Highest Frequency: {highest_freq:.2f} Hz\n'
                    f'   Lowest Frequency: {lowest_freq:.2f} Hz\n'
                    f'   Average Frequency: {average_freq:.2f} Hz', fontsize=12)
plt.title('Explanation and Frequency Analysis')
plt.axis('off')  # Turn off the axis for better readability

plt.tight_layout()
plt.show()

# Create a new figure for the explanation
plt.figure(figsize=(10, 5))
plt.text(0.0, 0.5, simplest_explain, fontsize=12, ha='left', va='center', wrap=True)
plt.title('Explanation of the Code 1D Fourier Transform')
plt.axis('off')  # Turn off the axis

# Adjust layout to accommodate text
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.show()


# Print the calculated frequency values
print(f"Highest Frequency: {highest_freq:.2f} Hz")
print(f"Lowest Frequency: {lowest_freq:.2f} Hz")
print(f"Average Frequency: {average_freq:.2f} Hz")
print(simplest_explain)




