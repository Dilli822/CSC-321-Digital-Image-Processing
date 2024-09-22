# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from urllib.request import urlopen

# # Load image from URL
# url = 'https://unsplash.it/444'
# image = np.asarray(bytearray(urlopen(url).read()), dtype="uint8")
# image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

# # Convert the image to binary using thresholding
# _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# # Show the original and binary images
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title('Original Image')
# plt.imshow(image, cmap='gray')
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.title('Binary Image')
# plt.imshow(binary_image, cmap='gray')
# plt.axis('off')

# plt.tight_layout()
# plt.show()

# def get_chain_code(contour):
#     chain_codes = []
#     for i in range(1, len(contour)):
#         diff = contour[i][0] - contour[i-1][0]
#         if np.array_equal(diff, [1, 0]):
#             chain_codes.append(0)  # Right
#         elif np.array_equal(diff, [1, -1]):
#             chain_codes.append(1)  # Top-Right
#         elif np.array_equal(diff, [0, -1]):
#             chain_codes.append(2)  # Top
#         elif np.array_equal(diff, [-1, -1]):
#             chain_codes.append(3)  # Top-Left
#         elif np.array_equal(diff, [-1, 0]):
#             chain_codes.append(4)  # Left
#         elif np.array_equal(diff, [-1, 1]):
#             chain_codes.append(5)  # Bottom-Left
#         elif np.array_equal(diff, [0, 1]):
#             chain_codes.append(6)  # Bottom
#         elif np.array_equal(diff, [1, 1]):
#             chain_codes.append(7)  # Bottom-Right
#     return chain_codes

# # Find contours of the binary image
# contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# # Apply chain code on the first contour
# chain_code = get_chain_code(contours[0])

# # Visualize the chain code
# plt.figure(figsize=(10, 5))
# plt.plot(chain_code)
# plt.title("Chain Code Representation")
# plt.xlabel("Point Index")
# plt.ylabel("Direction Code")
# plt.show()

# def compute_signature(contour):
#     moments = cv2.moments(contour)
    
#     if moments['m00'] != 0:
#         cx = int(moments['m10'] / moments['m00'])  # Centroid x
#         cy = int(moments['m01'] / moments['m00'])  # Centroid y
#     else:
#         cx, cy = 0, 0  # Fallback to (0, 0) if moments['m00'] is zero

#     distances = [np.sqrt((point[0][0] - cx) ** 2 + (point[0][1] - cy) ** 2) for point in contour]
#     return distances

# def shape_number(chain_code):
#     n = len(chain_code)
#     rotations = [chain_code[i:] + chain_code[:i] for i in range(n)]
#     return min(rotations)

# # Compute the shape number from chain codes
# shape_num = shape_number(chain_code)

# # Visualize the shape number
# plt.figure(figsize=(10, 5))
# plt.plot(shape_num)
# plt.title("Shape Number")
# plt.xlabel("Index")
# plt.ylabel("Code")
# plt.show()

# def fourier_descriptors(contour):
#     contour_complex = np.array([complex(p[0][0], p[0][1]) for p in contour])
#     fourier_result = np.fft.fft(contour_complex)
#     return fourier_result

# # Compute Fourier Descriptors
# fd = fourier_descriptors(contours[0])

# # Visualize Fourier Descriptors
# plt.figure(figsize=(10, 5))
# plt.plot(np.abs(fd))
# plt.title("Fourier Descriptors")
# plt.xlabel("Frequency")
# plt.ylabel("Magnitude")
# plt.show()

# # Print results
# print("Chain Code:", chain_code)
# print("Shape Number:", shape_num)
# print("Number of Fourier Descriptors:", len(fd))



# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from urllib.request import urlopen

# # Load and process image
# url = 'https://unsplash.it/444'
# image = np.asarray(bytearray(urlopen(url).read()), dtype="uint8")
# image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
# _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# # Find contours
# contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# main_contour = max(contours, key=cv2.contourArea)

# def get_chain_code(contour):
#     chain_codes = []
#     for i in range(1, len(contour)):
#         diff = contour[i][0] - contour[i-1][0]
#         if np.array_equal(diff, [1, 0]): chain_codes.append(0)     # Right
#         elif np.array_equal(diff, [1, -1]): chain_codes.append(1)  # Top-Right
#         elif np.array_equal(diff, [0, -1]): chain_codes.append(2)  # Top
#         elif np.array_equal(diff, [-1, -1]): chain_codes.append(3) # Top-Left
#         elif np.array_equal(diff, [-1, 0]): chain_codes.append(4)  # Left
#         elif np.array_equal(diff, [-1, 1]): chain_codes.append(5)  # Bottom-Left
#         elif np.array_equal(diff, [0, 1]): chain_codes.append(6)   # Bottom
#         elif np.array_equal(diff, [1, 1]): chain_codes.append(7)   # Bottom-Right
#     return chain_codes

# def compute_signature(contour):
#     moments = cv2.moments(contour)
#     cx = int(moments['m10'] / moments['m00']) if moments['m00'] != 0 else 0
#     cy = int(moments['m01'] / moments['m00']) if moments['m00'] != 0 else 0
#     distances = [np.sqrt((point[0][0] - cx)**2 + (point[0][1] - cy)**2) for point in contour]
#     return distances, (cx, cy)

# def fourier_descriptors(contour):
#     contour_complex = np.array([complex(p[0][0], p[0][1]) for p in contour])
#     fourier_result = np.fft.fft(contour_complex)
#     return fourier_result

# # Compute descriptors
# chain_code = get_chain_code(main_contour)
# signature, centroid = compute_signature(main_contour)
# fd = fourier_descriptors(main_contour)

# # Visualization functions
# def plot_image(img, title, cmap='gray'):
#     plt.figure(figsize=(10, 8))
#     plt.imshow(img, cmap=cmap)
#     plt.title(title)
#     plt.axis('off')
#     plt.show()

# def plot_descriptor(data, title, xlabel, ylabel, attribute):
#     plt.figure(figsize=(12, 6))
#     plt.plot(data)
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.text(0.05, 0.95, f'Attribute: {attribute}', transform=plt.gca().transAxes, verticalalignment='top')
#     plt.grid(True)
#     plt.show()

# # Plot original and binary images
# plot_image(image, 'Original Image')
# plot_image(binary_image, 'Binary Image')

# # Plot Chain Code
# plot_descriptor(chain_code, 'Chain Code', 'Contour Point Index', 'Direction Code', 'Boundary Direction')

# # Plot Shape Signature
# plot_descriptor(signature, 'Shape Signature', 'Contour Point Index', 'Distance from Centroid', 'Shape Variation')

# # Plot Fourier Descriptors
# plot_descriptor(np.abs(fd), 'Fourier Descriptors', 'Frequency', 'Magnitude', 'Shape Frequency Components')

# # Plot Contour with Centroid
# plt.figure(figsize=(10, 8))
# plt.imshow(binary_image, cmap='gray')
# plt.plot(main_contour[:, 0, 0], main_contour[:, 0, 1], 'r', linewidth=2)
# plt.plot(centroid[0], centroid[1], 'bo', markersize=10)
# plt.title('Contour with Centroid')
# plt.axis('off')
# plt.show()

# # Print summary
# print("Chain Code: Extracts the direction of the boundary. Each number (0-7) represents a direction.")
# print("Shape Signature: Represents the distance of each boundary point from the centroid.")
# print("Fourier Descriptors: Capture the frequency components of the shape, useful for shape analysis and recognition.")

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from urllib.request import urlopen

# # Load and process image
# url = 'https://unsplash.it/444'
# image = np.asarray(bytearray(urlopen(url).read()), dtype="uint8")
# image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
# _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# # Find contours
# contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# main_contour = max(contours, key=cv2.contourArea)

# def get_chain_code(contour):
#     chain_codes = []
#     for i in range(1, len(contour)):
#         diff = contour[i][0] - contour[i-1][0]
#         if np.array_equal(diff, [1, 0]): chain_codes.append(0)     # Right
#         elif np.array_equal(diff, [1, -1]): chain_codes.append(1)  # Top-Right
#         elif np.array_equal(diff, [0, -1]): chain_codes.append(2)  # Top
#         elif np.array_equal(diff, [-1, -1]): chain_codes.append(3) # Top-Left
#         elif np.array_equal(diff, [-1, 0]): chain_codes.append(4)  # Left
#         elif np.array_equal(diff, [-1, 1]): chain_codes.append(5)  # Bottom-Left
#         elif np.array_equal(diff, [0, 1]): chain_codes.append(6)   # Bottom
#         elif np.array_equal(diff, [1, 1]): chain_codes.append(7)   # Bottom-Right
#     return chain_codes

# def compute_signature(contour):
#     moments = cv2.moments(contour)
#     cx = int(moments['m10'] / moments['m00']) if moments['m00'] != 0 else 0
#     cy = int(moments['m01'] / moments['m00']) if moments['m00'] != 0 else 0
#     distances = [np.sqrt((point[0][0] - cx)**2 + (point[0][1] - cy)**2) for point in contour]
#     return distances, (cx, cy)

# def fourier_descriptors(contour):
#     contour_complex = np.array([complex(p[0][0], p[0][1]) for p in contour])
#     fourier_result = np.fft.fft(contour_complex)
#     return fourier_result

# # Compute descriptors
# chain_code = get_chain_code(main_contour)
# signature, centroid = compute_signature(main_contour)
# fd = fourier_descriptors(main_contour)

# # Visualization functions
# def plot_image(img, title, cmap='gray'):
#     plt.figure(figsize=(10, 8))
#     plt.imshow(img, cmap=cmap)
#     plt.title(title)
#     plt.axis('off')
#     plt.show()

# def plot_descriptor(data, title, xlabel, ylabel, attribute):
#     plt.figure(figsize=(12, 6))
#     plt.plot(data)
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.text(0.05, 0.95, f'Attribute: {attribute}', transform=plt.gca().transAxes, verticalalignment='top')
#     plt.grid(True)
#     plt.show()

# # Plot original and binary images
# plot_image(image, 'Original Image')
# plot_image(binary_image, 'Binary Image')

# # Plot Chain Code
# plot_descriptor(chain_code, 'Chain Code', 'Contour Point Index', 'Direction Code', 'Boundary Direction')

# # Plot Shape Signature
# plot_descriptor(signature, 'Shape Signature', 'Contour Point Index', 'Distance from Centroid', 'Shape Variation')

# # Plot Fourier Descriptors
# plot_descriptor(np.abs(fd), 'Fourier Descriptors', 'Frequency', 'Magnitude', 'Shape Frequency Components')

# # Plot Contour with Centroid
# plt.figure(figsize=(10, 8))
# plt.imshow(binary_image, cmap='gray')
# plt.plot(main_contour[:, 0, 0], main_contour[:, 0, 1], 'r', linewidth=2)
# plt.plot(centroid[0], centroid[1], 'bo', markersize=10)
# plt.title('Contour with Centroid')
# plt.axis('off')
# plt.show()

# # New: Combined plot of all descriptors
# plt.figure(figsize=(15, 12))

# # Original image with contour
# plt.subplot(2, 2, 1)
# plt.imshow(image, cmap='gray')
# plt.plot(main_contour[:, 0, 0], main_contour[:, 0, 1], 'r', linewidth=2)
# plt.plot(centroid[0], centroid[1], 'bo', markersize=10)
# plt.title('Original Image with Contour and Centroid')
# plt.axis('off')

# # Chain Code
# plt.subplot(2, 2, 2)
# plt.plot(chain_code)
# plt.title('Chain Code')
# plt.xlabel('Contour Point Index')
# plt.ylabel('Direction Code')

# # Shape Signature
# plt.subplot(2, 2, 3)
# plt.plot(signature)
# plt.title('Shape Signature')
# plt.xlabel('Contour Point Index')
# plt.ylabel('Distance from Centroid')

# # Fourier Descriptors
# plt.subplot(2, 2, 4)
# plt.plot(np.abs(fd))
# plt.title('Fourier Descriptors')
# plt.xlabel('Frequency')
# plt.ylabel('Magnitude')

# plt.tight_layout()
# plt.show()

# # Print summary
# print("Chain Code: Extracts the direction of the boundary. Each number (0-7) represents a direction.")
# print("Shape Signature: Represents the distance of each boundary point from the centroid.")
# print("Fourier Descriptors: Capture the frequency components of the shape, useful for shape analysis and recognition.")


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from urllib.request import urlopen

# # Load and process image
# url = 'https://unsplash.it/444'
# image = np.asarray(bytearray(urlopen(url).read()), dtype="uint8")
# image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
# _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# # Find contours
# contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# main_contour = max(contours, key=cv2.contourArea)

# def get_chain_code(contour):
#     chain_codes = []
#     for i in range(1, len(contour)):
#         diff = contour[i][0] - contour[i-1][0]
#         if np.array_equal(diff, [1, 0]): chain_codes.append(0)     # Right
#         elif np.array_equal(diff, [1, -1]): chain_codes.append(1)  # Top-Right
#         elif np.array_equal(diff, [0, -1]): chain_codes.append(2)  # Top
#         elif np.array_equal(diff, [-1, -1]): chain_codes.append(3) # Top-Left
#         elif np.array_equal(diff, [-1, 0]): chain_codes.append(4)  # Left
#         elif np.array_equal(diff, [-1, 1]): chain_codes.append(5)  # Bottom-Left
#         elif np.array_equal(diff, [0, 1]): chain_codes.append(6)   # Bottom
#         elif np.array_equal(diff, [1, 1]): chain_codes.append(7)   # Bottom-Right
#     return chain_codes

# def compute_signature(contour):
#     moments = cv2.moments(contour)
#     cx = int(moments['m10'] / moments['m00']) if moments['m00'] != 0 else 0
#     cy = int(moments['m01'] / moments['m00']) if moments['m00'] != 0 else 0
#     distances = [np.sqrt((point[0][0] - cx)**2 + (point[0][1] - cy)**2) for point in contour]
#     return distances, (cx, cy)

# def fourier_descriptors(contour):
#     contour_complex = np.array([complex(p[0][0], p[0][1]) for p in contour])
#     fourier_result = np.fft.fft(contour_complex)
#     return fourier_result

# # Compute descriptors
# chain_code = get_chain_code(main_contour)
# signature, centroid = compute_signature(main_contour)
# fd = fourier_descriptors(main_contour)

# # Reconstruction functions
# def reconstruct_from_chain_code(chain_code, start_point):
#     directions = [(1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1), (0,1), (1,1)]
#     contour = [start_point]
#     for code in chain_code:
#         x, y = contour[-1]
#         dx, dy = directions[code]
#         contour.append((x + dx, y + dy))
#     return np.array(contour)

# def reconstruct_from_signature(signature, centroid):
#     angles = np.linspace(0, 2*np.pi, len(signature), endpoint=False)
#     x = centroid[0] + signature * np.cos(angles)
#     y = centroid[1] + signature * np.sin(angles)
#     return np.column_stack((x, y)).astype(int)

# def reconstruct_from_fourier(fd, num_descriptors=10):
#     contour_complex = np.fft.ifft(fd[:num_descriptors])
#     contour = np.array([(int(z.real), int(z.imag)) for z in contour_complex])
#     return contour

# # Visualization function
# def plot_reconstructions(original, chain, signature, fourier):
#     fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    
#     axs[0, 0].imshow(original, cmap='gray')
#     axs[0, 0].set_title('Original Image')
#     axs[0, 0].axis('off')
    
#     axs[0, 1].imshow(np.zeros_like(original), cmap='gray')
#     axs[0, 1].plot(chain[:, 0], chain[:, 1], 'r')
#     axs[0, 1].set_title('Chain Code Reconstruction')
#     axs[0, 1].axis('off')
    
#     axs[1, 0].imshow(np.zeros_like(original), cmap='gray')
#     axs[1, 0].plot(signature[:, 0], signature[:, 1], 'g')
#     axs[1, 0].set_title('Shape Signature Reconstruction')
#     axs[1, 0].axis('off')
    
#     axs[1, 1].imshow(np.zeros_like(original), cmap='gray')
#     axs[1, 1].plot(fourier[:, 0], fourier[:, 1], 'b')
#     axs[1, 1].set_title('Fourier Descriptor Reconstruction')
#     axs[1, 1].axis('off')
    
#     plt.tight_layout()
#     plt.show()

# # Reconstruct shapes from descriptors
# chain_reconstruction = reconstruct_from_chain_code(chain_code, main_contour[0][0])
# signature_reconstruction = reconstruct_from_signature(signature, centroid)
# fourier_reconstruction = reconstruct_from_fourier(fd)

# # Plot reconstructions
# plot_reconstructions(image, chain_reconstruction, signature_reconstruction, fourier_reconstruction)

# # Print summary
# print("Chain Code Reconstruction: Shows the shape rebuilt using only directional information.")
# print("Shape Signature Reconstruction: Demonstrates the shape rebuilt using distances from the centroid.")
# print("Fourier Descriptor Reconstruction: Illustrates the shape rebuilt using frequency components.")




import cv2
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlopen

# Load and process image
url = 'https://unsplash.it/444'
image = np.asarray(bytearray(urlopen(url).read()), dtype="uint8")
image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
_, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
main_contour = max(contours, key=cv2.contourArea)

def get_chain_code(contour):
    chain_codes = []
    for i in range(1, len(contour)):
        diff = contour[i][0] - contour[i-1][0]
        if np.array_equal(diff, [1, 0]): chain_codes.append(0)     # Right
        elif np.array_equal(diff, [1, -1]): chain_codes.append(1)  # Top-Right
        elif np.array_equal(diff, [0, -1]): chain_codes.append(2)  # Top
        elif np.array_equal(diff, [-1, -1]): chain_codes.append(3) # Top-Left
        elif np.array_equal(diff, [-1, 0]): chain_codes.append(4)  # Left
        elif np.array_equal(diff, [-1, 1]): chain_codes.append(5)  # Bottom-Left
        elif np.array_equal(diff, [0, 1]): chain_codes.append(6)   # Bottom
        elif np.array_equal(diff, [1, 1]): chain_codes.append(7)   # Bottom-Right
    return chain_codes

def compute_signature(contour):
    moments = cv2.moments(contour)
    cx = int(moments['m10'] / moments['m00']) if moments['m00'] != 0 else 0
    cy = int(moments['m01'] / moments['m00']) if moments['m00'] != 0 else 0
    distances = [np.sqrt((point[0][0] - cx)**2 + (point[0][1] - cy)**2) for point in contour]
    angles = [np.arctan2(point[0][1] - cy, point[0][0] - cx) for point in contour]
    return distances, angles, (cx, cy)

def fourier_descriptors(contour):
    contour_complex = np.array([complex(p[0][0], p[0][1]) for p in contour])
    fourier_result = np.fft.fft(contour_complex)
    return fourier_result

# Compute descriptors
chain_code = get_chain_code(main_contour)
signature_distances, signature_angles, centroid = compute_signature(main_contour)
fd = fourier_descriptors(main_contour)

# Reconstruction functions
def reconstruct_from_chain_code(chain_code, start_point):
    directions = [(1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1), (0,1), (1,1)]
    contour = [start_point]
    for code in chain_code:
        x, y = contour[-1]
        dx, dy = directions[code]
        contour.append((x + dx, y + dy))
    return np.array(contour)

def reconstruct_from_signature(distances, angles, centroid):
    x = centroid[0] + distances * np.cos(angles)
    y = centroid[1] + distances * np.sin(angles)
    return np.column_stack((x, y)).astype(int)

def reconstruct_from_fourier(fd):
    contour_complex = np.fft.ifft(fd)
    contour = np.array([(int(z.real), int(z.imag)) for z in contour_complex])
    return contour

# Visualization function
def plot_reconstructions(original, chain, signature, fourier):
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    
    axs[0, 0].imshow(original, cmap='gray')
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(np.zeros_like(original), cmap='gray')
    axs[0, 1].plot(chain[:, 0], chain[:, 1], 'r')
    axs[0, 1].set_title('Chain Code Reconstruction')
    axs[0, 1].axis('off')
    
    axs[1, 0].imshow(np.zeros_like(original), cmap='gray')
    axs[1, 0].plot(signature[:, 0], signature[:, 1], 'g')
    axs[1, 0].set_title('Shape Signature Reconstruction')
    axs[1, 0].axis('off')
    
    axs[1, 1].imshow(np.zeros_like(original), cmap='gray')
    axs[1, 1].plot(fourier[:, 0], fourier[:, 1], 'b')
    axs[1, 1].set_title('Fourier Descriptor Reconstruction')
    axs[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Reconstruct shapes from descriptors
chain_reconstruction = reconstruct_from_chain_code(chain_code, main_contour[0][0])
signature_reconstruction = reconstruct_from_signature(signature_distances, signature_angles, centroid)
fourier_reconstruction = reconstruct_from_fourier(fd)

# Plot reconstructions
plot_reconstructions(image, chain_reconstruction, signature_reconstruction, fourier_reconstruction)

# Print summary
print("Chain Code Reconstruction: Perfectly rebuilds the shape using directional information.")
print("Shape Signature Reconstruction: Exactly reproduces the shape using distances and angles from the centroid.")
print("Fourier Descriptor Reconstruction: Precisely reconstructs the shape using all frequency components.")