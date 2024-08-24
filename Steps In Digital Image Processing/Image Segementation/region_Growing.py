import numpy as np
from scipy import ndimage

def region_growing(image, seed, threshold):
    """
    Region growing algorithm
    
    :param image: Input image (2D numpy array)
    :param seed: Seed point (tuple of y, x coordinates)
    :param threshold: Intensity threshold for region growing
    :return: Segmented binary image
    """
    rows, cols = image.shape
    segmented = np.zeros((rows, cols), dtype=np.uint8)
    segmented[seed] = 1
    seed_value = image[seed]
    
    def _get_neighbors(y, x):
        return [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]
    
    stack = [seed]
    while stack:
        y, x = stack.pop()
        for ny, nx in _get_neighbors(y, x):
            if 0 <= ny < rows and 0 <= nx < cols:
                if segmented[ny, nx] == 0 and abs(int(image[ny, nx]) - int(seed_value)) <= threshold:
                    segmented[ny, nx] = 1
                    stack.append((ny, nx))
    
    return segmented


import cv2
# image = cv2.imread('Kirat_King_(_Yalamber_)_Sankhuwasabha,_Nepal.jpg', 0)  # Read as grayscale
image = cv2.imread('naruto.jpeg', 0)  # Read as grayscale
seed = (100, 100)  # Example seed point (y, x)
threshold = 50
result = region_growing(image, seed, threshold)
cv2.imwrite('Rsegmented.png', result * 255)