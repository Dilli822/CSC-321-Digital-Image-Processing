import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image

def ideal_low_pass_filter(shape, cutoff):
    rows, cols = shape
    center = (rows / 2, cols / 2)
    r = np.sqrt((np.arange(rows) - center[0])[:, None]**2 + (np.arange(cols) - center[1])**2)
    H = np.zeros((rows, cols))
    H[r <= cutoff] = 1
    return H

def gaussian_low_pass_filter(shape, cutoff):
    rows, cols = shape
    center = (rows / 2, cols / 2)
    r = np.sqrt((np.arange(rows) - center[0])[:, None]**2 + (np.arange(cols) - center[1])**2)
    H = np.exp(-(r**2) / (2 * (cutoff / 2.0)**2))
    return H

def butterworth_low_pass_filter(shape, cutoff, order=2):
    rows, cols = shape
    center = (rows / 2, cols / 2)
    r = np.sqrt((np.arange(rows) - center[0])[:, None]**2 + (np.arange(cols) - center[1])**2)
    H = 1 / (1 + (r / cutoff)**(2 * order))
    return H

def apply_filter(image, filter_func, cutoff, order=2):
    # Convert image to frequency domain
    image_fft = np.fft.fft2(image)
    image_fft_shift = np.fft.fftshift(image_fft)
    
    # Create filter
    if filter_func == ideal_low_pass_filter:
        filter_mask = filter_func(image.shape, cutoff)
    elif filter_func == gaussian_low_pass_filter:
        filter_mask = filter_func(image.shape, cutoff)
    else:
        filter_mask = filter_func(image.shape, cutoff, order)
    
    # Apply filter
    filtered_fft_shift = image_fft_shift * filter_mask
    
    # Convert back to spatial domain
    filtered_fft = np.fft.ifftshift(filtered_fft_shift)
    filtered_image = np.fft.ifft2(filtered_fft)
    filtered_image = np.abs(filtered_image).astype(np.uint8)
    
    return filtered_image

# Fetch the image from URL
url = 'https://unsplash.it/555'  # Replace with your image URL
response = requests.get(url)
image_bytes = BytesIO(response.content)

# Convert to OpenCV format
image_pil = Image.open(image_bytes).convert('RGB')
image = np.array(image_pil)
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale

# Apply Ideal Low-Pass Filter
cutoff_frequency = 30
ideal_filtered_image = apply_filter(image, ideal_low_pass_filter, cutoff_frequency)

# Apply Gaussian Low-Pass Filter
gaussian_filtered_image = apply_filter(image, gaussian_low_pass_filter, cutoff_frequency)

# Apply Butterworth Low-Pass Filter
butterworth_filtered_image = apply_filter(image, butterworth_low_pass_filter, cutoff_frequency, order=2)

# Display results
cv2.imshow('Original Image', image)
cv2.imshow('Ideal Low-Pass Filter', ideal_filtered_image)
cv2.imshow('Gaussian Low-Pass Filter', gaussian_filtered_image)
cv2.imshow('Butterworth Low-Pass Filter', butterworth_filtered_image)

# Wait for a key press and close windows
print("Press any key to close the images...")
cv2.waitKey(0)
cv2.destroyAllWindows()
