import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

# Step 1: Load the image from a URL
url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRKNhBJdJa2FyGdf3UZEdtZsW5vHELQTWaBvw&s'  # Replace with your image URL
response = requests.get(url)
original_image = Image.open(BytesIO(response.content)).convert('L')  # Convert to grayscale

# Step 2: Resize the image to 8x8 pixels for pattern visualization
resized_image = original_image.resize((8, 8))

# Step 3: Convert the resized image to a NumPy array (matrix)
image_matrix = np.array(resized_image)
print("Resized Image Matrix:\n", image_matrix)

# Step 4: Flatten the matrix into a pattern vector
pattern_vector = image_matrix.flatten()
print("\nPattern Vector:\n", pattern_vector)

# Step 5: Create an 8x8 matrix from the pattern vector to visualize the vector pattern
vector_pattern_matrix = pattern_vector.reshape((8, 8))

# Step 6: Plot the original, vector pattern, and intensity pattern
plt.figure(figsize=(15, 5))

# Display the original image (scaled to 8x8 for visual consistency)
plt.subplot(1, 3, 1)
plt.imshow(original_image, cmap='gray', interpolation='nearest')
plt.title('Original Image')
plt.axis('off')

# Display the resized 8x8 image
plt.subplot(1, 3, 2)
plt.imshow(resized_image, cmap='gray', interpolation='nearest')
plt.title('Vector Pattern Image (8x8)')
plt.axis('off')

# Display the heatmap of pixel intensities
plt.subplot(1, 3, 3)
plt.imshow(vector_pattern_matrix, cmap='hot', interpolation='nearest')
plt.title('Pixel Intensity Pattern')
plt.colorbar(label='Intensity')
plt.grid(False)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

# Step 1: Load the image from a URL
url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRKNhBJdJa2FyGdf3UZEdtZsW5vHELQTWaBvw&s'  # Replace with your image URL
response = requests.get(url)
original_image = Image.open(BytesIO(response.content)).convert('L')  # Convert to grayscale

# Step 2: Resize the image to 8x8 pixels for pattern visualization
resized_image = original_image.resize((8, 8))

# Step 3: Convert the resized image to a NumPy array (matrix)
image_matrix = np.array(original_image.resize((8, 8)))  # Original image matrix
print("Resized Image Matrix (8x8):\n", image_matrix)

# Step 4: Flatten the matrix into a pattern vector
pattern_vector = image_matrix.flatten()
print("\nPattern Vector:\n", pattern_vector)

# Step 5: Create an 8x8 matrix from the pattern vector to visualize the vector pattern
vector_pattern_matrix = pattern_vector.reshape((8, 8))

# Step 6: Plot the original image matrix and vector pattern matrix
plt.figure(figsize=(12, 6))

# Plot the original image matrix
plt.subplot(1, 2, 1)
plt.imshow(image_matrix, cmap='gray', interpolation='nearest')
plt.title('Original Image Matrix')
plt.colorbar(label='Intensity')
plt.grid(False)

# Plot the vector pattern matrix
plt.subplot(1, 2, 2)
plt.imshow(vector_pattern_matrix, cmap='hot', interpolation='nearest')
plt.title('Vector Pattern Matrix')
plt.colorbar(label='Intensity')
plt.grid(False)

plt.tight_layout()
plt.show()

import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops

# =================== Shape Features ===================

def compute_shape_features(image_path):
    # Load the image (assumes binary or thresholded image)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Thresholding to get a binary image
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Find contours of the shapes
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Compute area and perimeter for compactness
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

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

        print(f"Shape Features: Compactness: {compactness}, Circularity: {circularity}, Eccentricity: {eccentricity}")


# =================== Moment Invariants ===================

def compute_moment_invariants(image_path):
    # Load the image (assumes binary or thresholded image)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Thresholding to get binary image
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Find contours of the shapes
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Compute moments of the contour
        moments = cv2.moments(contour)
        
        # Compute Hu moments (which are invariant under rotation, scaling, translation)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Print the moment invariants
        print(f"Moment Invariants: {hu_moments}")


# =================== Texture Features ===================

def compute_texture_features(image_path):
    # Load the image for texture analysis (grayscale)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Compute Gray Level Co-occurrence Matrix (GLCM)
    glcm = greycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    # Extract texture features
    contrast = greycoprops(glcm, 'contrast')[0, 0]
    correlation = greycoprops(glcm, 'correlation')[0, 0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
    
    # Entropy = sum(p * log2(p))
    entropy = -np.sum(glcm * np.log2(glcm + 1e-10))  # Add small epsilon to avoid log(0)

    print(f"Texture Features: Contrast: {contrast}, Correlation: {correlation}, Homogeneity: {homogeneity}, Entropy: {entropy}")


# =================== Full Feature Extraction ===================

def extract_features(image_path):
    print("===== Shape Features =====")
    compute_shape_features(image_path)
    
    print("\n===== Moment Invariants =====")
    compute_moment_invariants(image_path)
    
    print("\n===== Texture Features =====")
    compute_texture_features(image_path)


# =================== Run the Feature Extraction ===================
if __name__ == "__main__":
    # Replace 'image.png' with the path to your input image
    image_path = 'fit_result.png'
    extract_features(image_path)
