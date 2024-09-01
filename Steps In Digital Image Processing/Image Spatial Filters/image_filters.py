
# BEST FOR CONTRAST IMAGE, CT SCAN, MEDICAL IMAGES, TEXT IMAGES AND SO ON... 
# FIRST ORDER DERIVATIVE OPERATORS DETECT EDGE AND HIGHLIGHT RAPID INTENSITY AREAS DETECTION 
import cv2
import numpy as np
from scipy.ndimage import convolve
import urllib.request
import tempfile
import os

def load_image_from_url(url):
    # Download image from URL and save it to a temporary file
    with urllib.request.urlopen(url) as response:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp_file.write(response.read())
        temp_file.close()
        return temp_file.name

def load_image(image_path):
    # Load image from the given path
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Optional: URL of the image
# image_url = "https://i0.wp.com/digital-photography-school.com/wp-content/uploads/2014/11/Horse.jpg?w=600&h=1260&ssl=1"  # Set this to your image URL or keep it as None for local file
# image_url = "https://media.springernature.com/lw685/springer-static/image/chp%3A10.1007%2F978-3-030-14828-7_1/MediaObjects/83014_2_En_1_Fig13_HTML.jpg"
image_url = "https://www.harleystreet.sg/img/cardiac-computed-tomography(1).webp"

if image_url:
    image_path = load_image_from_url(image_url)
else:
    image_path = 'space.jpg'  # Local file path

# Load an image
image = load_image(image_path)

# Define operator kernels
roberts_gx = np.array([[1, 0], [0, -1]], dtype='int')
roberts_gy = np.array([[0, 1], [-1, 0]], dtype='int')

sobel_gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='int')
sobel_gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype='int')

prewitt_gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype='int')
prewitt_gy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype='int')

# Function to apply filter and compute magnitude
def apply_filter(image, gx, gy):
    g_x = convolve(image, gx)
    g_y = convolve(image, gy)
    magnitude = np.sqrt(g_x**2 + g_y**2)
    
    # Normalize to range [0, 255]
    magnitude = np.uint8(255 * magnitude / np.max(magnitude))
    return magnitude

# Apply Roberts operator
roberts_output = apply_filter(image, roberts_gx, roberts_gy)
cv2.imshow('Roberts Operator', roberts_output)

# Apply Sobel operator
sobel_output = apply_filter(image, sobel_gx, sobel_gy)
cv2.imshow('Sobel Operator', sobel_output)

# Apply Prewitt operator
prewitt_output = apply_filter(image, prewitt_gx, prewitt_gy)
cv2.imshow('Prewitt Operator', prewitt_output)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Cleanup temporary file if used
if image_url:
    os.remove(image_path)

