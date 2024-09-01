import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import convolve

# Function to generate a synthetic image matrix
def generate_synthetic_image(size=(100, 100)):
    image = np.zeros(size)
    # Create a pattern or shape
    image[20:40, 20:40] = 255  # White square
    image[60:80, 60:80] = 255  # Another white square
    # Add a gradient
    gradient = np.linspace(0, 255, size[1])
    image[40:60, :] = gradient
    return image

# Generate a synthetic image
image = generate_synthetic_image()

# Define operators
def define_operators():
    # Roberts operator kernels
    roberts_gx = np.array([[1, 0], [0, -1]])
    roberts_gy = np.array([[0, 1], [-1, 0]])
    
    # Sobel operator kernels
    sobel_gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    # Prewitt operator kernels
    prewitt_gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_gy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    
    return roberts_gx, roberts_gy, sobel_gx, sobel_gy, prewitt_gx, prewitt_gy

# Function to apply filter and compute magnitude
def apply_filter(image, gx, gy):
    g_x = convolve(image, gx)
    g_y = convolve(image, gy)
    magnitude = np.sqrt(g_x**2 + g_y**2)
    
    # Normalize to range [0, 255]
    magnitude = np.uint8(255 * magnitude / np.max(magnitude))
    return magnitude

# Apply filters
roberts_gx, roberts_gy, sobel_gx, sobel_gy, prewitt_gx, prewitt_gy = define_operators()

roberts_output = apply_filter(image, roberts_gx, roberts_gy)
sobel_output = apply_filter(image, sobel_gx, sobel_gy)
prewitt_output = apply_filter(image, prewitt_gx, prewitt_gy)

# Function to display scatter plot and 3D plot
def visualize_results(image, roberts_output, sobel_output, prewitt_output):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Scatter plot of the original image
    axs[0, 0].scatter(*np.indices(image.shape), c=image.flatten(), cmap='gray')
    axs[0, 0].set_title('Original Image')
    
    # Scatter plot of Roberts output
    axs[0, 1].scatter(*np.indices(roberts_output.shape), c=roberts_output.flatten(), cmap='gray')
    axs[0, 1].set_title('Roberts Operator')
    
    # Scatter plot of Sobel output
    axs[1, 0].scatter(*np.indices(sobel_output.shape), c=sobel_output.flatten(), cmap='gray')
    axs[1, 0].set_title('Sobel Operator')
    
    # Scatter plot of Prewitt output
    axs[1, 1].scatter(*np.indices(prewitt_output.shape), c=prewitt_output.flatten(), cmap='gray')
    axs[1, 1].set_title('Prewitt Operator')
    
    plt.tight_layout()
    plt.show()
    
    # 3D plots
    fig = plt.figure(figsize=(18, 6))
    
    # Original image 3D plot
    ax = fig.add_subplot(131, projection='3d')
    X, Y = np.indices(image.shape)
    ax.plot_surface(X, Y, image, cmap='gray')
    ax.set_title('Original Image 3D')
    
    # Roberts output 3D plot
    ax = fig.add_subplot(132, projection='3d')
    ax.plot_surface(X, Y, roberts_output, cmap='gray')
    ax.set_title('Roberts Operator 3D')
    
    # Sobel output 3D plot
    ax = fig.add_subplot(133, projection='3d')
    ax.plot_surface(X, Y, sobel_output, cmap='gray')
    ax.set_title('Sobel Operator 3D')
    
    plt.tight_layout()
    plt.show()

# Visualize results
visualize_results(image, roberts_output, sobel_output, prewitt_output)
