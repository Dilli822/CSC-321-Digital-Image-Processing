"""
 - can identify if an image is too dark, too bright, or lacks contrast.
 - to improve the contrast of an image by redistributing intensity values. 
 - It spreads out the most frequent intensity values, enhancing details in both bright and dark areas of the image.
 - to determine thresholds for segmenting an image into regions or objects, especially in binary image processing where a clear separation of intensity levels is needed.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image

def manual_histogram_equalization_from_url(image_url):
    # Download the image from the URL
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert('L')  # Convert to grayscale
    img = np.array(img)

    # Get the image dimensions
    height, width = img.shape
    total_pixels = height * width
    
    # Compute the histogram of the input image
    histogram = np.zeros(256)
    for pixel in img.ravel():
        histogram[pixel] += 1

    # Probability Density Function (PDF)
    pdf = histogram / total_pixels

    # Cumulative Distribution Function (CDF)
    cdf = np.cumsum(pdf)

    # Transformation function based on CDF
    equalized_levels = np.round(cdf * 255).astype(np.uint8)

    # Apply the equalization mapping to the original image
    equalized_img = equalized_levels[img]

    # Plot original and equalized histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(img.ravel(), bins=256, color='gray', alpha=0.7)
    ax1.set_title('Original Histogram')
    ax1.set_xlabel('Pixel Intensity')
    ax1.set_ylabel('Frequency')

    ax2.hist(equalized_img.ravel(), bins=256, color='blue', alpha=0.7)
    ax2.set_title('Equalized Histogram')
    ax2.set_xlabel('Pixel Intensity')
    ax2.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    # Display the original and equalized images using matplotlib
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(equalized_img, cmap='gray')
    plt.title('Equalized Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# image_url = 'https://unsplash.it/450'
image_url = "https://prod-images-static.radiopaedia.org/images/35899381/3bb7c333a3e81a2bda64a9275335a9_gallery.jpeg"
 # Replace with the URL of your image
manual_histogram_equalization_from_url(image_url)
