import cv2
import numpy as np
from matplotlib import pyplot as plt

def apply_8_way_laplacian(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if the image was loaded properly
    if image is None:
        print(f"Error: Could not load the image at {image_path}. Please check the file path and try again.")
        return
    
    # Define the 8-way Laplacian kernel
    kernel_8_way = np.array([[1, 1, 1],
                             [1, -8, 1],
                             [1, 1, 1]])
    
    # Apply the Laplacian filter using 8-way connectivity
    laplacian_8_way = cv2.filter2D(image, -1, kernel_8_way)
    
    # Create subplots with adjustable size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={'width_ratios': [1, 1.5]}) # Adjust width_ratios and height_ratios as needed
    
    # Original Image
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # 8-Way Laplacian Image
    ax2.imshow(laplacian_8_way, cmap='gray')
    ax2.set_title('8-Way Laplacian')
    ax2.axis('off')
    
    # Adjust layout (optional)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)  # Adjust width and height spacing between subplots
    plt.show()

# Replace with your correct image path
image_path = 'space.jpg'  # Ensure the file exists here
apply_8_way_laplacian(image_path)
