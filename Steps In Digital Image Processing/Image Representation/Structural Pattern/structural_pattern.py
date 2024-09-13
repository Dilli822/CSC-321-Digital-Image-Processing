"""
The code helps to analyze and describe various properties of the objects and textures within an image using a 
combination of shape, statistical, and texture analysis techniques. This kind of feature extraction is often 
used in image recognition, computer vision, and machine learning tasks to better understand and classify images 
based on their visual patterns.

"""

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # =================== Shape Features ===================
# def compute_shape_features(image):
#     # Thresholding to get a binary image
#     _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

#     # Find contours of the shapes
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     shape_features = []

#     for contour in contours:
#         # Compute area and perimeter for compactness
#         area = cv2.contourArea(contour)
#         perimeter = cv2.arcLength(contour, True)

#         if area == 0:
#             continue  # Skip contours with zero area

#         # Compactness = (Perimeter^2) / (4π * Area)
#         compactness = (perimeter ** 2) / (4 * np.pi * area)

#         # Fit an ellipse to compute eccentricity
#         if len(contour) >= 5:  # FitEllipse needs at least 5 points
#             ellipse = cv2.fitEllipse(contour)
#             major_axis = max(ellipse[1])
#             minor_axis = min(ellipse[1])
#             eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
#         else:
#             eccentricity = 0

#         # Circularity = 4π(Area) / (Perimeter^2)
#         circularity = (4 * np.pi * area) / (perimeter ** 2)

#         shape_features.append({
#             'contour': contour,
#             'compactness': compactness,
#             'circularity': circularity,
#             'eccentricity': eccentricity
#         })

#     return shape_features

# # =================== Moment Invariants ===================
# def compute_moment_invariants(image):
#     # Thresholding to get binary image
#     _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

#     # Find contours of the shapes
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     hu_moments = []
#     for contour in contours:
#         # Compute moments of the contour
#         moments = cv2.moments(contour)
#         # Compute Hu moments (which are invariant under rotation, scaling, translation)
#         hu = cv2.HuMoments(moments).flatten()
#         hu_moments.append(hu)

#     return hu_moments

# # =================== Texture Features (without skimage) ===================
# def compute_glcm(image, distances, angles):
#     """ 
#     Manually compute Gray Level Co-occurrence Matrix (GLCM).
#     This assumes the image is quantized into 256 levels.
#     """
#     max_gray = 256
#     glcm = np.zeros((max_gray, max_gray), dtype=np.uint32)

#     rows, cols = image.shape
#     for row in range(rows):
#         for col in range(cols):
#             current_pixel = image[row, col]
#             for d, angle in zip(distances, angles):
#                 dx = int(np.round(d * np.cos(angle)))
#                 dy = int(np.round(d * np.sin(angle)))

#                 # Get coordinates of neighbor pixel
#                 neighbor_row = row + dy
#                 neighbor_col = col + dx

#                 if 0 <= neighbor_row < rows and 0 <= neighbor_col < cols:
#                     neighbor_pixel = image[neighbor_row, neighbor_col]
#                     glcm[current_pixel, neighbor_pixel] += 1

#     return glcm

# def compute_texture_features(image):
#     # Manually compute the GLCM (Gray Level Co-occurrence Matrix)
#     glcm = compute_glcm(image, distances=[1], angles=[0])

#     # Normalize the GLCM
#     glcm = glcm / np.sum(glcm)

#     # Compute texture features
#     contrast = np.sum([[(i - j) ** 2 * glcm[i, j] for i in range(256)] for j in range(256)])
#     correlation = np.sum([[(i * j * glcm[i, j]) for i in range(256)] for j in range(256)]) - np.mean(glcm)
#     homogeneity = np.sum([[glcm[i, j] / (1.0 + abs(i - j)) for i in range(256)] for j in range(256)])
#     entropy = -np.sum(glcm * np.log2(glcm + 1e-10))  # Add small epsilon to avoid log(0)

#     return {
#         'contrast': contrast,
#         'correlation': correlation,
#         'homogeneity': homogeneity,
#         'entropy': entropy
#     }

# # =================== Visualization for Each Feature ===================

# def plot_shape_features(image, shape_features):
#     # Visualize contours with eccentricity annotations
#     image_with_contours = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#     for shape in shape_features:
#         cv2.drawContours(image_with_contours, [shape['contour']], -1, (0, 255, 0), 2)
#         # Annotate with eccentricity
#         M = cv2.moments(shape['contour'])
#         if M["m00"] != 0:
#             cX = int(M["m10"] / M["m00"])
#             cY = int(M["m01"] / M["m00"])
#             # cv2.putText(image_with_contours, f"Ecc: {shape['eccentricity']:.2f}", (cX, cY),
#             #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


#     plt.figure(figsize=(8, 6))
#     plt.imshow(image_with_contours)
#     plt.title('Shape Features (Contours & Eccentricity)')
#     plt.axis('off')
#     plt.show()


# def plot_moment_invariants(hu_moments):
#     # Visualize Hu moments as bar plot
#     hu_moments_mean = np.mean(hu_moments, axis=0)
    
#     plt.figure(figsize=(8, 6))
#     plt.bar(range(1, 8), np.abs(hu_moments_mean), color='blue')
#     plt.title('Moment Invariants (Hu Moments)')
#     plt.xlabel('Hu Moments')
#     plt.ylabel('Value')
#     plt.show()


# def plot_texture_features(texture_features):
#     # Visualize Texture Features as bar plot
#     texture_labels = ['Contrast', 'Correlation', 'Homogeneity', 'Entropy']
#     texture_values = list(texture_features.values())
    
#     plt.figure(figsize=(8, 6))
#     plt.bar(texture_labels, texture_values, color='orange')
#     plt.title('Texture Features')
#     plt.ylabel('Value')
#     plt.show()


# # =================== Full Feature Extraction and Visualization ===================
# def visualize_features(image_path):
#     # Load the image (grayscale)
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#     # Shape Features
#     shape_features = compute_shape_features(image)
#     plot_shape_features(image, shape_features)
    
#     # Moment Invariants
#     hu_moments = compute_moment_invariants(image)
#     plot_moment_invariants(hu_moments)
    
#     # Texture Features
#     texture_features = compute_texture_features(image)
#     plot_texture_features(texture_features)

# # =================== Run the Visualization ===================
# if __name__ == "__main__":
#     image_path = 'fit_result.png'  # Replace with the actual path of your image
#     visualize_features(image_path)



import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image

# Function to load image from a URL or local path
def load_image(image_path_or_url):
    if image_path_or_url.startswith('http'):  # Check if the input is a URL
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content)).convert('L')  # Convert to grayscale
        image = np.array(image)
    else:
        image = cv2.imread(image_path_or_url, cv2.IMREAD_GRAYSCALE)
    return image

# =================== Shape Features ===================
def compute_shape_features(image):
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shape_features = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area == 0:
            continue  # Skip contours with zero area

        # Compactness = (Perimeter^2) / (4π * Area)
        compactness = (perimeter ** 2) / (4 * np.pi * area)

        # Fit an ellipse to compute eccentricity
        if len(contour) >= 5:
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

# =================== Moment Invariants ===================
def compute_moment_invariants(image):
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hu_moments = []
    for contour in contours:
        moments = cv2.moments(contour)
        hu = cv2.HuMoments(moments).flatten()
        hu_moments.append(hu)

    return hu_moments

# =================== Texture Features (without skimage) ===================
def compute_glcm(image, distances, angles):
    max_gray = 256
    glcm = np.zeros((max_gray, max_gray), dtype=np.uint32)

    rows, cols = image.shape
    for row in range(rows):
        for col in range(cols):
            current_pixel = image[row, col]
            for d, angle in zip(distances, angles):
                dx = int(np.round(d * np.cos(angle)))
                dy = int(np.round(d * np.sin(angle)))

                neighbor_row = row + dy
                neighbor_col = col + dx

                if 0 <= neighbor_row < rows and 0 <= neighbor_col < cols:
                    neighbor_pixel = image[neighbor_row, neighbor_col]
                    glcm[current_pixel, neighbor_pixel] += 1

    return glcm

def compute_texture_features(image):
    glcm = compute_glcm(image, distances=[1], angles=[0])
    glcm = glcm / np.sum(glcm)

    contrast = np.sum([[(i - j) ** 2 * glcm[i, j] for i in range(256)] for j in range(256)])
    correlation = np.sum([[(i * j * glcm[i, j]) for i in range(256)] for j in range(256)]) - np.mean(glcm)
    homogeneity = np.sum([[glcm[i, j] / (1.0 + abs(i - j)) for i in range(256)] for j in range(256)])
    entropy = -np.sum(glcm * np.log2(glcm + 1e-10))

    return {
        'contrast': contrast,
        'correlation': correlation,
        'homogeneity': homogeneity,
        'entropy': entropy
    }

# =================== Visualization for Each Feature ===================
def plot_shape_features(image, shape_features):
    image_with_contours = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for shape in shape_features:
        cv2.drawContours(image_with_contours, [shape['contour']], -1, (0, 255, 0), 2)
        M = cv2.moments(shape['contour'])
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

    plt.figure(figsize=(8, 6))
    plt.imshow(image_with_contours)
    plt.title('Shape Features (Contours & Eccentricity)')
    plt.axis('off')
    plt.show()

def plot_moment_invariants(hu_moments):
    hu_moments_mean = np.mean(hu_moments, axis=0)
    
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, 8), np.abs(hu_moments_mean), color='blue')
    plt.title('Moment Invariants (Hu Moments)')
    plt.xlabel('Hu Moments')
    plt.ylabel('Value')
    plt.show()

def plot_texture_features(texture_features):
    texture_labels = ['Contrast', 'Correlation', 'Homogeneity', 'Entropy']
    texture_values = list(texture_features.values())
    
    plt.figure(figsize=(8, 6))
    plt.bar(texture_labels, texture_values, color='orange')
    plt.title('Texture Features')
    plt.ylabel('Value')
    plt.show()

# =================== Full Feature Extraction and Visualization ===================
def visualize_features(image_path_or_url):
    image = load_image(image_path_or_url)

    # Shape Features
    shape_features = compute_shape_features(image)
    plot_shape_features(image, shape_features)
    
    # Moment Invariants
    hu_moments = compute_moment_invariants(image)
    plot_moment_invariants(hu_moments)
    
    # Texture Features
    texture_features = compute_texture_features(image)
    plot_texture_features(texture_features)

# =================== Run the Visualization ===================
if __name__ == "__main__":
    image_path_or_url = 'https://img.auntminnieeurope.com/files/base/smg/all/image/2012/01/ame.2012_01_12_13_10_50_17_2012_01_xx_fetalMRI.png?auto=format%2Ccompress&fit=max&q=70&w=400'  # Replace with your image URL or local path
    visualize_features(image_path_or_url)
