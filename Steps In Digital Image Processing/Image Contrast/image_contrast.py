import numpy as np
import matplotlib.pyplot as plt

def contrast_stretching(image, r1, r2, with_background=False):
    """
    Apply contrast stretching to the input image.
    
    :param image: Input image as a 2D numpy array
    :param r1: Lower threshold
    :param r2: Upper threshold
    :param with_background: If True, preserve background; if False, apply clipping
    :return: Contrast stretched image
    """
    output = np.zeros_like(image)
    
    if with_background:
        # With background
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if r1 <= image[i, j] <= r2:
                    output[i, j] = 7  # L-1 = 7 (assuming 8-bit image)
                else:
                    output[i, j] = image[i, j]
    else:
        # Without background (clipping)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if r1 <= image[i, j] <= r2:
                    output[i, j] = 7  # L-1 = 7 (assuming 8-bit image)
                else:
                    output[i, j] = 0
    
    return output

def plot_images(original, without_bg, with_bg):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.imshow(original, cmap='gray', vmin=0, vmax=7)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(without_bg, cmap='gray', vmin=0, vmax=7)
    ax2.set_title('Without Background')
    ax2.axis('off')
    
    ax3.imshow(with_bg, cmap='gray', vmin=0, vmax=7)
    ax3.set_title('With Background')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage
input_image = np.array([
    [4, 3, 5, 2],
    [3, 6, 4, 6],
    [2, 2, 6, 5],
    [7, 6, 4, 1]
])

r1, r2 = 3, 5

# Without background (clipping)
output_without_bg = contrast_stretching(input_image, r1, r2, with_background=False)
print("Output image without background:")
print(output_without_bg)

# With background
output_with_bg = contrast_stretching(input_image, r1, r2, with_background=True)
print("\nOutput image with background:")
print(output_with_bg)

# Visualize the results
plot_images(input_image, output_without_bg, output_with_bg)