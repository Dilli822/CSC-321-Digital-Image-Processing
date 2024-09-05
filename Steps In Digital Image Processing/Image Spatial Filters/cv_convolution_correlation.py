import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d, correlate2d
import requests
from io import BytesIO

def load_image(image_path):
    """
    Load an image from a URL or a local path.
    """
    try:
        # Try loading from URL
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert('L')  # Convert to grayscale
        print("Loaded image from URL.")
    except Exception as e:
        # Fallback to local path if URL fails
        print(f"Failed to load image from URL: {e}. Attempting to load from local path.")
        image = Image.open(image_path).convert('L')  # Convert to grayscale
    return np.array(image)

# Define image path (replace with your URL or local path)
image_path = "https://unsplash.it/555"  # Replace with a valid URL or a local path like 'path/to/image.jpg'

# Load the image
image = load_image(image_path)

# Define a kernel (2x2 example)
kernel = np.array([
    [0, 1],
    [-1, 0]
])

# Perform Convolution
convolved_image = convolve2d(image, kernel, mode='same')

# Perform Correlation
correlated_image = correlate2d(image, kernel, mode='same')

# Display Results
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(convolved_image, cmap='gray')
plt.title('Convolved Image')

plt.subplot(1, 3, 3)
plt.imshow(correlated_image, cmap='gray')
plt.title('Correlated Image')

plt.tight_layout()
plt.show()

