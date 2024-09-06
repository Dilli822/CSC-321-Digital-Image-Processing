"""
- detecting isolated points or outliers by highlighting pixels that are 
significantly different from their surroundings.
- images
1. Thermal or Infrared Images
2. medical images
3. astronomical images
4. microscopy images
 -- Do not use for these images ---
1. Images with Uniform Patterns:
2. High-Detail Color Images
"""
import cv2
import numpy as np
import urllib.request
from matplotlib import pyplot as plt

def load_image(image_path_or_url):
    # Check if the input is a URL
    if image_path_or_url.startswith('http'):
        resp = urllib.request.urlopen(image_path_or_url)
        image_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path_or_url, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError("Image could not be loaded. Check the path or URL.")
    return image

def isolated_point_detection(image, threshold=8):
    # Define the kernel
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    
    # Apply the filter using the kernel
    filtered_image = cv2.filter2D(image, -1, kernel)
    
    # Apply the threshold to create a binary output
    binary_output = np.where(np.abs(filtered_image) > threshold, 1, 0)
    
    return filtered_image, binary_output

# Example usage
# image_path_or_url = 'https://images.hothardware.com/contentimages/newsitem/60943/content/galaxy-ngc-7496.jpg'  # Replace with local path or URL
# image_path_or_url = "https://unsplash.it/550"
# image_path_or_url = "https://i.kinja-img.com/image/upload/c_fill,h_900,q_60,w_1600/c49cf5c459d0d6c17351018cfa64610f.jpg"
# image_path_or_url = "https://butlerandland.com/wp-content/uploads/2021/12/Thermal-Imaging-Camera-Deer.jpg"
# image_path_or_url = "https://img.freepik.com/free-photo/people-colorful-thermal-scan-with-celsius-degree-temperature_23-2149170127.jpg"
image_path_or_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTJo29p93o9CI5lDm8xbjKraY2RnGWwT7BdKg&s"
image = load_image(image_path_or_url)

# Detect isolated points
filtered, binary = isolated_point_detection(image, threshold=25)

# Adjust the figure size to make images larger
plt.figure(figsize=(10, 8))  # Increased figure size

# Display the original image
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')  # Hide axes to focus on the image

# Display the filtered image
plt.subplot(1, 3, 2)
plt.title('Filtered Image')
plt.imshow(filtered, cmap='gray')
plt.axis('off')  # Hide axes

# Display the binary output
plt.subplot(1, 3, 3)
plt.title('Binary Output')
plt.imshow(binary, cmap='gray')
plt.axis('off')  # Hide axes

plt.tight_layout()  # Adjust spacing between subplots for a cleaner look
plt.show()
