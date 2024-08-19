import numpy as np
import cv2

# Path to the text file containing the binary matrix
input_file_path = './binary_image_matrix.txt'
output_image_path = 'reconstructed_image.jpg'

# Read the binary matrix from the text file
with open(input_file_path, 'r') as file:
    lines = file.readlines()

# Extract the matrix
matrix = []
for line in lines[1:]:  # Skip the header line
    # Convert each line of text to a list of integers
    row = [int(value) for value in line.split()]
    matrix.append(row)

# Convert the list of lists to a NumPy array
binary_matrix = np.array(matrix, dtype=np.uint8)

# Save the binary matrix as an image
cv2.imwrite(output_image_path, binary_matrix)

# Optionally display the image
cv2.imshow('Reconstructed Image', binary_matrix)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Reconstructed image has been saved to {output_image_path}")
