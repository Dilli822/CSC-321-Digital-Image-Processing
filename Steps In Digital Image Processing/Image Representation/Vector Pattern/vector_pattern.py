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


