import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

# Function for contrast stretching using the original formulas from the image notes
def contrast_stretching(img, r1=3, r2=5, s1=2, s2=6):
    # Create an empty output image
    output_img = np.zeros(img.shape, dtype=np.uint8)
    
    # Compute the alpha, beta, and gamma values
    alpha = s1 / r1
    beta = (s2 - s1) / (r2 - r1)
    gamma = (7 - s2) / (7 - r2)
    
    # Apply piecewise linear transformation based on input ranges
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            r = img[i, j]  # pixel value

            # Apply the corresponding formula depending on the value of r
            if 0 <= r < r1:
                output_img[i, j] = alpha * r
            elif r1 <= r < r2:
                output_img[i, j] = beta * (r - r1) + s1
            elif r2 <= r <= 7:
                output_img[i, j] = gamma * (r - r2) + s2
            else:
                output_img[i, j] = r  # No change for values outside [0, 7] range
            
    return output_img

# Function to fetch an image from a URL
def fetch_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

# URL of the image (you can replace this with any online image)
url = 'https://www.frontiersin.org/files/Articles/252866/fped-05-00036-HTML/image_m/fped-05-00036-g001.jpg' 
url = 'https://i.pinimg.com/736x/2e/87/6c/2e876c62684bf1911614ef6d0d0bce4f.jpg'
url = 'https://storage.googleapis.com/fm-coresites-assets/mpora_new/wp-content/uploads/2016/01/Infrared-Thermal-Imaging-Avalanche-Victims-680x510.jpg'
url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQjh5B2IFnWjADWsE2qNcRowv6IYNomyNQaKg&s'

# Fetch the image
img = fetch_image_from_url(url)

# Convert the image to grayscale using PIL
img_gray = img.convert('L')

# Convert the grayscale image to a NumPy array
img_np = np.array(img_gray)

# Normalize the pixel values to fit the [0,7] range as per the notes
normalized_img = np.interp(img_np, (img_np.min(), img_np.max()), (0, 7))

# Apply the contrast stretching
stretched_img = contrast_stretching(normalized_img)

# Normalize the output image back to [0, 255] for display purposes
stretched_img_normalized = np.interp(stretched_img, (0, 7), (0, 255)).astype(np.uint8)

# Plot the original and contrast stretched images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img_np, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Contrast Stretched Image")
plt.imshow(stretched_img_normalized, cmap='gray')

plt.show()
