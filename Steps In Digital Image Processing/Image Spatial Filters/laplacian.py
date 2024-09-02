import cv2
import numpy as np
from matplotlib import pyplot as plt

def apply_4_way_laplacian(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Define the 4-way Laplacian kernel
    kernel_4_way = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]])
    
    # Apply the Laplacian filter using 4-way connectivity
    laplacian_4_way = cv2.filter2D(image, -1, kernel_4_way)
    
    # Plot the original and filtered images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('4-Way Laplacian')
    plt.imshow(laplacian_4_way, cmap='gray')
    plt.show()

# Replace with your image path or URL
image_path = 'space.jpg'
apply_4_way_laplacian(image_path)
