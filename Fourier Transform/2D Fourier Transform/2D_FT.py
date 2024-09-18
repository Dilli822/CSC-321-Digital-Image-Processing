import numpy as np
import matplotlib.pyplot as plt
from skimage import data

# Load the built-in grayscale image
image = data.camera()

# Apply 2D Fourier Transform
F = np.fft.fft2(image)
Fshift = np.fft.fftshift(F)  # Shift the zero frequency component to the center

# Calculate the different components
real_part = np.real(Fshift)
imaginary_part = np.imag(Fshift)
magnitude_spectrum = np.log(np.abs(Fshift) + 1)  # Magnitude spectrum
phase_spectrum = np.angle(Fshift)  # Phase spectrum

# Plot the original image and the Fourier Transform components
plt.figure(figsize=(12, 10))

# Plot the original image
plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Plot the real part of the Fourier Transform
plt.subplot(2, 3, 2)
plt.imshow(real_part, cmap='gray')
plt.title('Real Part')
plt.axis('off')

# Plot the imaginary part of the Fourier Transform
plt.subplot(2, 3, 3)
plt.imshow(imaginary_part, cmap='gray')
plt.title('Imaginary Part')
plt.axis('off')

# Plot the magnitude spectrum
plt.subplot(2, 3, 4)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum')
plt.axis('off')

# Plot the phase spectrum
plt.subplot(2, 3, 5)
plt.imshow(phase_spectrum, cmap='gray')
plt.title('Phase Spectrum')
plt.axis('off')

# Plot a combined log-magnitude and phase
plt.subplot(2, 3, 6)
combined_image = np.log(np.abs(Fshift)) * np.angle(Fshift)
plt.imshow(combined_image, cmap='gray')
plt.title('Log Magnitude * Phase')
plt.axis('off')

plt.tight_layout()
plt.show()
