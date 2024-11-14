import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Load a real image
image = cv2.imread('lab3Image.png', cv2.IMREAD_GRAYSCALE)

# Load a real image
imagee = cv2.imread('lab3_1Image.jpeg', cv2.IMREAD_GRAYSCALE)

# Ensure the image was loaded correctly
if image is None or imagee is None:
    raise ValueError("Image not loaded. Ensure the image file path is correct.")

# 1. Point Detection using Laplacian (discontinuity based)
laplacian = cv2.Laplacian(image, cv2.CV_64F)
laplacian_abs = cv2.convertScaleAbs(laplacian)

# 2. Line Detection using Sobel operator (Gradient-based edge detection)
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel_edge = cv2.magnitude(sobel_x, sobel_y)
sobel_edge = cv2.convertScaleAbs(sobel_edge)

# 3. Mexican Hat filter (Second derivative of Gaussian)
def mexican_hat_filter(image, sigma=1.0):
    gaussian = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.Laplacian(gaussian, cv2.CV_64F)

mexican_hat = mexican_hat_filter(image)

# 4. Edge Linking and Boundary Detection using Canny edge detector
edges = cv2.Canny(image, 100, 200)

# Plot results of the edge detection methods
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(laplacian_abs, cmap='gray')
plt.title('Point Detection (Laplacian)')

plt.subplot(2, 3, 2)
plt.imshow(sobel_edge, cmap='gray')
plt.title('Line Detection (Sobel)')

plt.subplot(2, 3, 3)
plt.imshow(mexican_hat, cmap='gray')
plt.title('Mexican Hat Filter')

plt.subplot(2, 3, 4)
plt.imshow(edges, cmap='gray')
plt.title('Edge Linking (Canny)')

# 5. Thresholding (Global, Local, Adaptive)
_, global_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
local_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

plt.subplot(2, 3, 5)
plt.imshow(global_thresh, cmap='gray')
plt.title('Global Thresholding')

plt.subplot(2, 3, 6)
plt.imshow(local_thresh, cmap='gray')
plt.title('Adaptive Thresholding')

plt.show()

