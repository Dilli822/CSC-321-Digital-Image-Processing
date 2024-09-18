import numpy as np
import matplotlib.pyplot as plt
import requests
from scipy.io import wavfile
from io import BytesIO
from scipy.signal import spectrogram

# Download and read the audio file
url = 'https://www2.cs.uic.edu/~i101/SoundFiles/StarWars60.wav'
response = requests.get(url)
audio_data = BytesIO(response.content)
sample_rate, audio_signal = wavfile.read(audio_data)

# Convert stereo to mono if needed
if audio_signal.ndim > 1:
    audio_signal = audio_signal[:, 0]

# Apply the Fourier Transform
F = np.fft.fft(audio_signal)
frequencies = np.fft.fftfreq(len(F), 1 / sample_rate)
magnitude_spectrum = np.abs(F)
phase_spectrum = np.angle(F)

# Set up the figure and axis grid
fig, axs = plt.subplots(3, 2, figsize=(14, 12))  # 3 rows x 2 columns grid
fig.suptitle('Audio Signal and Frequency Analysis', fontsize=16)

# Plot 1: Raw Audio Signal (Time Domain)
axs[0, 0].plot(audio_signal)
axs[0, 0].set_title('Raw Audio Signal (Time Domain)')
axs[0, 0].set_xlabel('Sample Index')
axs[0, 0].set_ylabel('Amplitude')
axs[0, 0].grid()

# Plot 2: Zoomed-In View of the Audio Signal (Time Domain)
axs[0, 1].plot(audio_signal[:5000])  # Zoom in on the first 5000 samples
axs[0, 1].set_title('Zoomed-In Audio Signal (Time Domain)')
axs[0, 1].set_xlabel('Sample Index')
axs[0, 1].set_ylabel('Amplitude')
axs[0, 1].grid()

# Plot 3: Magnitude Spectrum (Frequency Domain)
axs[1, 0].plot(frequencies[:len(frequencies)//2], magnitude_spectrum[:len(frequencies)//2])
axs[1, 0].set_title('Magnitude Spectrum (Frequency Domain)')
axs[1, 0].set_xlabel('Frequency (Hz)')
axs[1, 0].set_ylabel('Magnitude')
axs[1, 0].grid()

# Plot 4: Zoomed-In View of the Magnitude Spectrum (Frequency Domain)
axs[1, 1].plot(frequencies[:1000], magnitude_spectrum[:1000])  # Zoom in on the first 1000 frequencies
axs[1, 1].set_title('Zoomed-In Magnitude Spectrum (Frequency Domain)')
axs[1, 1].set_xlabel('Frequency (Hz)')
axs[1, 1].set_ylabel('Magnitude')
axs[1, 1].grid()

# Plot 5: Phase Spectrum
axs[2, 0].plot(frequencies[:len(frequencies)//2], phase_spectrum[:len(frequencies)//2])
axs[2, 0].set_title('Phase Spectrum')
axs[2, 0].set_xlabel('Frequency (Hz)')
axs[2, 0].set_ylabel('Phase (Radians)')
axs[2, 0].grid()

# Plot 6: Spectrogram
frequencies_spec, times_spec, Sxx = spectrogram(audio_signal, sample_rate)
cax = axs[2, 1].pcolormesh(times_spec, frequencies_spec, 10 * np.log10(Sxx), shading='gouraud')
axs[2, 1].set_title('Spectrogram')
axs[2, 1].set_xlabel('Time (s)')
axs[2, 1].set_ylabel('Frequency (Hz)')
fig.colorbar(cax, ax=axs[2, 1], label='Intensity (dB)')
axs[2, 1].grid()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to make space for the suptitle
plt.show()




# Create a new figure for explanations
fig, axs = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('Explaining Each Plot', fontsize=16)

# Plot 1 Explanation
axs[0, 0].plot(audio_signal)
axs[0, 0].set_title('Plot 1: Raw Audio Signal')
axs[0, 0].set_xlabel('Sample Index')
axs[0, 0].set_ylabel('Amplitude')
axs[0, 0].text(0.5, 0.5, 'This plot shows how the sound changes over time.\nImagine it like waves going up and down.', 
                horizontalalignment='center', verticalalignment='center', 
                transform=axs[0, 0].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
axs[0, 0].grid()

# Plot 2 Explanation
axs[0, 1].plot(audio_signal[:5000])
axs[0, 1].set_title('Plot 2: Zoomed-In Audio Signal')
axs[0, 1].set_xlabel('Sample Index')
axs[0, 1].set_ylabel('Amplitude')
axs[0, 1].text(0.5, 0.5, 'Here we zoom in on a small part of the sound.\nIt helps us see details better, like looking at a small section of waves.', 
                horizontalalignment='center', verticalalignment='center', 
                transform=axs[0, 1].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
axs[0, 1].grid()

# Plot 3 Explanation
axs[1, 0].plot(frequencies[:len(frequencies)//2], magnitude_spectrum[:len(frequencies)//2])
axs[1, 0].set_title('Plot 3: Magnitude Spectrum')
axs[1, 0].set_xlabel('Frequency (Hz)')
axs[1, 0].set_ylabel('Magnitude')
axs[1, 0].text(0.5, 0.5, 'This plot shows which frequencies are in the sound and how strong they are.\nThink of it like a list of all the notes played.', 
                horizontalalignment='center', verticalalignment='center', 
                transform=axs[1, 0].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
axs[1, 0].grid()

# Plot 4 Explanation
axs[1, 1].plot(frequencies[:1000], magnitude_spectrum[:1000])
axs[1, 1].set_title('Plot 4: Zoomed-In Magnitude Spectrum')
axs[1, 1].set_xlabel('Frequency (Hz)')
axs[1, 1].set_ylabel('Magnitude')
axs[1, 1].text(0.5, 0.5, 'This zooms in on the first part of the frequency list.\nIt helps us see the smaller details of the frequencies better.', 
                horizontalalignment='center', verticalalignment='center', 
                transform=axs[1, 1].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
axs[1, 1].grid()

# Plot 5 Explanation
axs[2, 0].plot(frequencies[:len(frequencies)//2], phase_spectrum[:len(frequencies)//2])
axs[2, 0].set_title('Plot 5: Phase Spectrum')
axs[2, 0].set_xlabel('Frequency (Hz)')
axs[2, 0].set_ylabel('Phase (Radians)')
axs[2, 0].text(0.5, 0.5, 'This plot shows the phase of each frequency.\nIt’s like knowing the starting point of each wave.', 
                horizontalalignment='center', verticalalignment='center', 
                transform=axs[2, 0].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
axs[2, 0].grid()

# Plot 6 Explanation
frequencies_spec, times_spec, Sxx = spectrogram(audio_signal, sample_rate)
cax = axs[2, 1].pcolormesh(times_spec, frequencies_spec, 10 * np.log10(Sxx), shading='gouraud')
axs[2, 1].set_title('Plot 6: Spectrogram')
axs[2, 1].set_xlabel('Time (s)')
axs[2, 1].set_ylabel('Frequency (Hz)')
fig.colorbar(cax, ax=axs[2, 1], label='Intensity (dB)')
axs[2, 1].text(0.5, 0.5, 'This shows how the frequencies change over time.\nImagine it like a movie of the sound waves.', 
                horizontalalignment='center', verticalalignment='center', 
                transform=axs[2, 1].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
axs[2, 1].grid()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to make space for the suptitle
plt.show()