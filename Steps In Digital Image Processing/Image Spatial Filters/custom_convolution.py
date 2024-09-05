
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
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

def custom_convolve2d(image, kernel):
    """
    Custom implementation of 2D convolution.
    """
    # Get the dimensions of the image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Flip the kernel (180 degrees)
    kernel = np.flipud(np.fliplr(kernel))
    
    # Calculate padding
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    # Pad the image with zeros
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    
    # Initialize the output array
    output = np.zeros_like(image)
    
    # Perform convolution
    for i in range(image_height):
        for j in range(image_width):
            # Extract the region of interest
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            # Element-wise multiplication and sum
            output[i, j] = np.sum(region * kernel)
    
    return output

# Define image path (replace with your URL or local path)
image_path = "https://unsplash.it/350"  # Replace with a valid URL or a local path like 'path/to/image.jpg'

# Load the image
image = load_image(image_path)

# Define a kernel (2x2 example)
kernel = np.array([
    [0, 1],
    [-1, 0]
])

# Perform Convolution using the built-in method
try:
    from scipy.signal import convolve2d
    convolved_image_builtin = convolve2d(image, kernel, mode='same')
except ImportError:
    print("SciPy is not installed. Please install it to use built-in convolution.")

# Perform Convolution using the custom method
convolved_image_custom = custom_convolve2d(image, kernel)

# Display Results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(convolved_image_custom, cmap='gray')
plt.title('Custom Convolved Image')

# Check if built-in function was used
if 'convolved_image_builtin' in locals():
    plt.subplot(1, 3, 3)
    plt.imshow(convolved_image_builtin, cmap='gray')
    plt.title('Built-in Convolved Image')

plt.tight_layout()
plt.show()

