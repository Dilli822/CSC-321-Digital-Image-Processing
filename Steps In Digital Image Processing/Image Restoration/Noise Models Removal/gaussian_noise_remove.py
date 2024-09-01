import cv2
import numpy as np
from matplotlib import pyplot as plt

def contraharmonic_mean_filter(image, kernel_size, Q):
    rows, cols = image.shape
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)
    filtered_image = np.zeros_like(image, dtype=np.float64)

    for i in range(rows):
        for j in range(cols):
            neighborhood = padded_image[i:i + kernel_size, j:j + kernel_size]
            numerator = np.sum(np.power(neighborhood, Q + 1))
            denominator = np.sum(np.power(neighborhood, Q))
            if denominator != 0:
                filtered_image[i, j] = numerator / denominator
            else:
                filtered_image[i, j] = 0

    filtered_image = np.clip(filtered_image, 0, 255)
    return filtered_image.astype(np.uint8)

def plot_frequency_spectrum(image, title):
    f_transform = np.fft.fft2(image)
    f_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shifted) + 1)
    
    plt.title(title)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.colorbar()

# Load the noisy image
image = cv2.imread('noisy_image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply the Contraharmonic Mean Filter
# use 3x3 , 5x5, 7x7
kernel_size = 3
# use negative for salt noise
Q = 2.0
filtered_image = contraharmonic_mean_filter(image, kernel_size, Q)

# Calculate noise levels (standard deviation)
noise_level_noisy = np.std(image)
noise_level_denoised = np.std(filtered_image)

# Display the original and denoised images with their frequency spectra
plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
plt.title('Original Noisy Image')
plt.imshow(image, cmap='gray')

plt.subplot(2, 2, 2)
plt.title('Denoised Image')
plt.imshow(filtered_image, cmap='gray')

plt.subplot(2, 2, 3)
plot_frequency_spectrum(image, 'Noisy Image Frequency Spectrum')

plt.subplot(2, 2, 4)
plot_frequency_spectrum(filtered_image, 'Denoised Image Frequency Spectrum')

plt.show()

# Print noise levels
print(f"Noise Level of Noisy Image: {noise_level_noisy}")
print(f"Noise Level of Denoised Image: {noise_level_denoised}")

# Determine which image is noisier
if noise_level_noisy > noise_level_denoised:
    print("The original image is noisier than the denoised image.")
else:
    print("The denoised image is noisier than the original image (unexpected).")
