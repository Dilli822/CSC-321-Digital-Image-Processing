"""
Architectural and Structural Photos
Road and Street Scenes
Grayscale or High-Contrast Images
Simple Diagrams and Blueprints
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

def read_image_from_url(url):
    """Reads an image from a given URL."""
    try:
        resp = urllib.request.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode the image.")
        return image
    except Exception as e:
        print(f"Error fetching image from URL: {e}")
        return None

def draw_lines(image, lines):
    """Draws lines on the image using the detected Hough lines."""
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))
            x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    else:
        print("No lines detected.")
    return image

def hough_line_transform(image):
    """Applies Hough Line Transform to detect lines."""
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    return lines

# Provide the URL of the image you want to process
# image_url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRI8ZaB0LYZP4DMCPcnAxZq6_FgnRFYTdUHtw&s'
# image_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS87v0EYFJPmZyTTSgkweCH-YdkIDw66ehwZTAis_nZbK2TVoFz-aELk1ABdgRhZyVEcrw&usqp=CAU"

# Load the image from the URL
img = read_image_from_url(image_url)

if img is not None:
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Hough Transform and draw lines
    lines = hough_line_transform(gray)
    result = draw_lines(img.copy(), lines)

    # Display results
    plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    plt.subplot(122), plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)), plt.title('Detected Lines')
    plt.show()
else:
    print("Image could not be loaded.")

