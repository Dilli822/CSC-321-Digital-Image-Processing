import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image  # Import the Image module from PIL

# =================== Shape Features ===================
def compute_shape_features(image):
    # Thresholding to get a binary image
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Find contours of the shapes
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shape_features = []

    for contour in contours:
        # Compute area and perimeter for compactness
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if area == 0:
            continue  # Skip contours with zero area

        # Compactness = (Perimeter^2) / (4π * Area)
        compactness = (perimeter ** 2) / (4 * np.pi * area)

        # Fit an ellipse to compute eccentricity
        if len(contour) >= 5:  # FitEllipse needs at least 5 points
            ellipse = cv2.fitEllipse(contour)
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
            eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
        else:
            eccentricity = 0

        # Circularity = 4π(Area) / (Perimeter^2)
        circularity = (4 * np.pi * area) / (perimeter ** 2)

        shape_features.append({
            'contour': contour,
            'compactness': compactness,
            'circularity': circularity,
            'eccentricity': eccentricity
        })

    return shape_features

def plot_shape_features(image, shape_features):
    # Visualize contours with eccentricity annotations
    image_with_contours = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for shape in shape_features:
        cv2.drawContours(image_with_contours, [shape['contour']], -1, (0, 255, 0), 2)
        # Annotate with eccentricity
        M = cv2.moments(shape['contour'])
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # cv2.putText(image_with_contours, f"Ecc: {shape['eccentricity']:.2f}", (cX, cY),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    plt.figure(figsize=(8, 6))
    plt.imshow(image_with_contours)
    plt.title('Shape Features (Contours & Eccentricity)')
    plt.axis('off')
    plt.show()

# =================== Load Image ===================
def load_image(image_path_or_url):
    if image_path_or_url.startswith('http'):
        # If it's a URL, download the image
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content)).convert('L')
        image = np.array(image)
    else:
        # Otherwise, assume it's a local file path
        image = cv2.imread(image_path_or_url, cv2.IMREAD_GRAYSCALE)
    return image

# =================== Full Feature Extraction and Visualization ===================
def visualize_shape_features(image_path_or_url):
    # Load the image
    image = load_image(image_path_or_url)

    # Shape Features
    shape_features = compute_shape_features(image)
    plot_shape_features(image, shape_features)

# =================== Run the Visualization ===================
if __name__ == "__main__":
    # image_path_or_url = 'https://prod-images-static.radiopaedia.org/images/54699894/22._gallery.jpeg'  
    image_path_or_url = 'https://www.vhlab.umn.edu/atlas/static-mri/4-chamber/graphics/startimage.jpg'
    visualize_shape_features(image_path_or_url)
