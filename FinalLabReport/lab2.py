import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load a real image (example: an image named 'example_image.jpg')
image = cv2.imread('lab1Img1.jpg', cv2.IMREAD_GRAYSCALE)

# Threshold the image to make it binary (black and white)
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Define a 3x3 kernel (structuring element)
kernel = np.ones((3, 3), np.uint8)

# Dilation: Expands the boundaries of foreground pixels (white pixels)
dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

# Erosion: Shrinks the boundaries of foreground pixels (white pixels)
eroded_image = cv2.erode(binary_image, kernel, iterations=1)

# Opening: Erosion followed by Dilation, removes noise (small white spots)
opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

# Closing: Dilation followed by Erosion, removes small black holes
closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

# Hit and Miss Operation (Custom Implementation using OpenCV functions)
hit_image = cv2.morphologyEx(binary_image, cv2.MORPH_HITMISS, kernel)

# Miss is essentially the inverse of Hit, so we can use the NOT of the result
miss_image = cv2.bitwise_not(hit_image)

# Fit is generally defined as the overlap of dilation and erosion
# So we calculate the intersection of dilation and erosion
fit_image = cv2.bitwise_and(dilated_image, eroded_image)

# Plotting the results
titles = ['Original Image', 'Binary Image', 'Dilated Image', 'Eroded Image', 'Opened Image', 'Closed Image', 'Hit Image', 'Miss Image', 'Fit Image']
images = [image, binary_image, dilated_image, eroded_image, opened_image, closed_image, hit_image, miss_image, fit_image]

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.show()
