import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('lab1Img1.jpg', cv2.IMREAD_GRAYSCALE)

# 1. Basic Gray Level Transformations

# Contrast Stretching
def contrast_stretching(img):
    min_val = np.min(img)
    max_val = np.max(img)
    stretched = ((img - min_val) / (max_val - min_val)) * 255
    return stretched.astype(np.uint8)

# Log Transformation
def log_transformation(img):
    c = 255 / np.log(1 + np.max(img))
    log_image = c * (np.log(1 + img))
    return log_image.astype(np.uint8)

# Digital Negative
negative_image = 255 - image

# Display Gray Level Transformations
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(contrast_stretching(image), cmap='gray')
axs[0].set_title('Contrast Stretching')
axs[1].imshow(log_transformation(image), cmap='gray')
axs[1].set_title('Log Transformation')
axs[2].imshow(negative_image, cmap='gray')
axs[2].set_title('Digital Negative')
plt.show()

# 2. Histogram Processing
# Histogram Equalization
equalized_image = cv2.equalizeHist(image)

# Display Histogram Equalization
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(122), plt.imshow(equalized_image, cmap='gray'), plt.title('Equalized Image')
plt.show()

# 3. Spatial Operations
# Applying Median Filter
median_filtered = cv2.medianBlur(image, 5)

# Applying Sobel Filter
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

# Applying Laplacian Filter
laplacian_filtered = cv2.Laplacian(image, cv2.CV_64F)

# Display Spatial Filters
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(median_filtered, cmap='gray')
axs[0].set_title('Median Filter')
axs[1].imshow(sobel_combined, cmap='gray')
axs[1].set_title('Sobel Filter')
axs[2].imshow(laplacian_filtered, cmap='gray')
axs[2].set_title('Laplacian Filter')
plt.show()

# 4. Magnification Techniques
# Replication
replicated_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)

# Interpolation
interpolated_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

# Display Magnification Results
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(replicated_image, cmap='gray')
axs[0].set_title('Magnification by Replication')
axs[1].imshow(interpolated_image, cmap='gray')
axs[1].set_title('Magnification by Interpolation')
plt.show()
