
"""
Morphological operations are techniques used in image processing that focus on the shape or structure of 
objects within an image.
"""

import cv2
import numpy as np
import urllib.request
import os

def load_image(path):
    """Load an image from a file or URL."""
    if path.startswith(('http://', 'https://')):
        try:
            with urllib.request.urlopen(path) as url:
                img_array = np.array(bytearray(url.read()), dtype=np.uint8)
                return cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        except urllib.error.URLError as e:
            print(f"Error loading image from URL: {e}")
            return None
    elif os.path.isfile(path):
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        print(f"Error: {path} is neither a valid URL nor an existing file.")
        return None

def hit_or_miss(image, kernel):
    """Perform hit-or-miss operation."""
    eroded = cv2.erode(image, kernel)
    dilated = cv2.dilate(image, 1 - kernel)
    return cv2.bitwise_and(eroded, cv2.bitwise_not(dilated))

def fit(image, kernel):
    """Perform fit operation."""
    return cv2.erode(image, kernel)

# Load the image (replace with your image path or URL)
# image_path = 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/ad/CT_Scan_General_Illustration.jpg/640px-CT_Scan_General_Illustration.jpg'
image_path = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ7fUAZMT3GUTs5OLRWnKCjoWivPH5RZZJCRw&s"
image = load_image(image_path)

if image is None:
    print("Failed to load the image. Exiting.")
    exit()

# Create the mask filter
mask = np.array([1, 0, 1, 1, 1, 0, 1, 0, 0], dtype=np.uint8).reshape(3, 3)

# Perform hit-or-miss operation
hit_miss_result = hit_or_miss(image, mask)

# Perform fit operation
fit_result = fit(image, mask)

# Display results
cv2.imshow('Original Image', image)
cv2.imshow('Hit-or-Miss Result', hit_miss_result)
cv2.imshow('Fit Result', fit_result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the results
cv2.imwrite('hit_miss_result.png', hit_miss_result)
cv2.imwrite('fit_result.png', fit_result)