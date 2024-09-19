
import numpy as np
import cv2
import matplotlib.pyplot as plt
from urllib.request import urlopen
import io
from PIL import Image

# Function to load an image from a URL and convert it to grayscale
def load_image_from_url(url):
    with urlopen(url) as response:
        image_data = response.read()
    img = Image.open(io.BytesIO(image_data)).convert('L')  # Convert to grayscale
    return np.array(img)

# Apply 2D Fourier Transform to the image
def apply_fourier_transform(image):
    # Apply 2D Fourier Transform
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    
    # Compute the magnitude spectrum and ensure no log of zero
    magnitude_spectrum = 20 * np.log(np.clip(np.abs(f_transform_shifted), a_min=1e-10, a_max=None))
    
    return magnitude_spectrum, f_transform_shifted

# Apply Inverse Fourier Transform to reconstruct the image
def apply_inverse_fourier_transform(f_transform_shifted):
    # Reverse the shift and apply the inverse FFT
    f_ishift = np.fft.ifftshift(f_transform_shifted)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    return img_back

# Visualize the original image and its Fourier transform results
def visualize_results(image, magnitude_spectrum):
    plt.figure(figsize=(12, 6))
    
    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")
    
    # Fourier Magnitude Spectrum
    plt.subplot(1, 2, 2)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title("Magnitude Spectrum")
    plt.axis("off")
    
    plt.show()

# Visualize low and high frequency components separately
def visualize_different_frequencies(image, f_transform_shifted):
    # Get image dimensions
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # Low frequencies (center part of the spectrum)
    low_freq_mask = np.zeros((rows, cols), np.uint8)
    low_freq_mask[crow-30:crow+30, ccol-30:ccol+30] = 1

    # Apply the low-frequency mask
    low_freq_transform = f_transform_shifted * low_freq_mask
    low_freq_image = apply_inverse_fourier_transform(low_freq_transform)

    # High frequencies (remove center part of the spectrum)
    high_freq_mask = 1 - low_freq_mask
    high_freq_transform = f_transform_shifted * high_freq_mask
    high_freq_image = apply_inverse_fourier_transform(high_freq_transform)

    # Plot the original, low-frequency, and high-frequency images
    plt.figure(figsize=(18, 6))

    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    # Low Frequency Image
    plt.subplot(1, 3, 2)
    plt.imshow(low_freq_image, cmap='gray')
    plt.title("Low Frequencies (Smooth Areas)")
    plt.axis('off')

    # High Frequency Image
    plt.subplot(1, 3, 3)
    plt.imshow(high_freq_image, cmap='gray')
    plt.title("High Frequencies (Sharp Details)")
    plt.axis('off')

    plt.show()

# URL of the image to process
image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/600px-PNG_transparency_demonstration_1.png'

# Load and process the image
image = load_image_from_url(image_url)
magnitude_spectrum, f_transform_shifted = apply_fourier_transform(image)

# Display the original and Fourier transform results
visualize_results(image, magnitude_spectrum)

# Visualize low and high frequency components
visualize_different_frequencies(image, f_transform_shifted)
