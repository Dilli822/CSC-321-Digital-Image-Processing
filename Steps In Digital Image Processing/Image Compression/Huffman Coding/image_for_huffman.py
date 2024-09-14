import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
import requests
from io import BytesIO
import gc
from tabulate import tabulate

# Function to load image (local path or URL)
def load_image(image_path):
    if image_path.startswith('http://') or image_path.startswith('https://'):
        response = requests.get(image_path)
        img = Image.open(BytesIO(response.content))
        img = np.array(img)
    else:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    return img

# Resize image for optimized memory usage
def resize_image(image, max_size=512):
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scaling_factor = max_size / float(max(h, w))
        new_size = (int(w * scaling_factor), int(h * scaling_factor))
        return cv2.resize(image, new_size)
    return image

# Extract pixel frequencies
def extract_pixel_frequencies(image):
    # Flatten the image and use a tuple (R, G, B) as keys
    pixels = image.reshape(-1, 3)
    pixel_freq = Counter(map(tuple, pixels))
    return pixel_freq


# Print the original matrix in the console
def print_image_matrix(image):
    print("Image Matrix (RGB values):")
    print(image)

# Count total number of bits in the image
def count_total_bits(image):
    h, w, c = image.shape
    total_pixels = h * w
    bits_per_pixel = 8 * c  # 8 bits per channel
    total_bits = total_pixels * bits_per_pixel
    return total_bits

# Count number of unique pixels
def count_unique_pixels(pixel_freq):
    return len(pixel_freq)

# Print a table of pixel frequencies and their bit sizes
def print_pixel_table(pixel_freq):
    table_data = []
    for pixel, freq in pixel_freq.items():
        # Convert pixel tuple to a string
        if isinstance(pixel, np.ndarray):
            pixel_str = ','.join(map(str, pixel))
        else:
            pixel_str = str(pixel)  # Handle cases where pixel is not an ndarray
        
        # Calculate bits required for the pixel
        bits = 8 * len(pixel)  # 8 bits per channel
        table_data.append([pixel_str, freq, bits])
    
    headers = ["Pixel (R,G,B)", "Frequency", "Bits"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


# Show the original matrix as a flat plot and the flattened matrix as text
def show_image_matrix_with_flattened(image):
    plt.figure(figsize=(8, 8))
    
    # Display the image
    plt.imshow(image)
    plt.title("Original Image Matrix (Flat Plot)")
    plt.axis('off')  # Hide axes for better visual appearance
    
    # Flatten the matrix and convert to a string
    flattened_matrix = image.flatten()
    matrix_str = np.array2string(flattened_matrix, separator=', ', threshold=10)  # Shorten for readability
    
    # Display the matrix string on the plot
    plt.text(0.5, -0.1, matrix_str, ha='center', va='top', fontsize=8, wrap=True, transform=plt.gca().transAxes)
    
    plt.show()

# Visualize 3D scatter plot (with random sampling to reduce memory usage)
def visualize_3d_scatter(image, sample_size=5000):
    h, w, _ = image.shape
    # Reshape the image and sample points for scatter plot
    rgb_values = image.reshape((h * w, 3))
    
    # Random sampling to reduce memory consumption
    if h * w > sample_size:
        indices = np.random.choice(np.arange(h * w), size=sample_size, replace=False)
        rgb_values = rgb_values[indices]
    
    # Extract R, G, B values
    r = rgb_values[:, 0]
    g = rgb_values[:, 1]
    b = rgb_values[:, 2]
    
    # 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(r, g, b, c=rgb_values / 255.0, marker='o', s=1)
    
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    
    plt.title("3D Scatter Plot of RGB Values")
    plt.show()

# Visualize 2D bar graph for pixel intensity frequencies
def visualize_2d_bar(pixel_freq):
    # Convert pixel frequencies into intensity values and their frequencies
    # Note: Pixel values are in tuples of (R, G, B)
    # Calculate intensity for each pixel by summing R, G, B values
    intensity_freq = Counter(np.sum(np.array(list(pixel_freq.keys())), axis=1))
    
    intensities = list(intensity_freq.keys())
    frequencies = list(intensity_freq.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(intensities, frequencies, color='gray')
    
    plt.xlabel('Pixel Intensity (Sum of R, G, B)')
    plt.ylabel('Frequency')
    plt.title('2D Bar Graph of Pixel Intensities')
    plt.show()


def main(image_path):
    # Load and resize the image for better memory efficiency
    image = load_image(image_path)
    image = resize_image(image)
    
    # Print the image matrix in the console
    print_image_matrix(image)
    
    # Show the original image matrix in a flat plot with flattened matrix as text
    show_image_matrix_with_flattened(image)
    
    # Extract pixel frequencies
    pixel_freq = extract_pixel_frequencies(image)

    # Print the table of pixel frequencies and bits
    print_pixel_table(pixel_freq)
    
    # Count total number of bits
    total_bits = count_total_bits(image)
    print(f"Total number of bits: {total_bits}")
    
    # Count number of unique pixels
    unique_pixels = count_unique_pixels(pixel_freq)
    print(f"Number of unique pixels: {unique_pixels}")
    # Visualize 2D bar graph for pixel intensities
    visualize_2d_bar(pixel_freq)
    
    # Visualize 3D scatter plot of RGB values
    visualize_3d_scatter(image)
    
    # Garbage collection to free memory
    gc.collect()

# Example usage
# image_path = 'https://images.pexels.com/photos/104827/cat-pet-animal-domestic-104827.jpeg'  
image_path = 'https://www.xtremeclimbers.com/assets/uploads/gallery/images/saipal-base-camp1559397748.jpg'
main(image_path)
