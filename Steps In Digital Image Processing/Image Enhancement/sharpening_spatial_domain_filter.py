import cv2
import numpy as np

def sharpen_image(image_path):
    # Read the input image
    img = cv2.imread(image_path)
    
    # Convert to grayscale if the image is color
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create the sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    
    # Apply the kernel to the image using convolution
    sharpened = cv2.filter2D(img, -1, kernel)
    
    # Clip the result to ensure pixel values are within [0, 255]
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    # Display the original and sharpened images
    cv2.imshow('Original Image', img)
    cv2.imshow('Sharpened Image', sharpened)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Optionally, save the sharpened image
    cv2.imwrite('sharpened_image.jpg', sharpened)

# Usage
sharpen_image('naruto.jpeg')