import numpy as np
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_ubyte
from skimage.measure import regionprops, label
from skimage.color import label2rgb
import cv2

def split_image(image, threshold):
    """
    Split the image into quadrants if not homogeneous
    
    :param image: Input image (2D numpy array)
    :param threshold: Homogeneity threshold
    :return: List of homogeneous regions
    """
    if image.size == 0 or np.max(image) - np.min(image) <= threshold:
        return [image]
    
    height, width = image.shape
    mid_h, mid_w = height // 2, width // 2
    
    top_left = split_image(image[:mid_h, :mid_w], threshold)
    top_right = split_image(image[:mid_h, mid_w:], threshold)
    bottom_left = split_image(image[mid_h:, :mid_w], threshold)
    bottom_right = split_image(image[mid_h:, mid_w:], threshold)
    
    return top_left + top_right + bottom_left + bottom_right

def merge_regions(image, regions, threshold):
    """
    Merge adjacent similar regions
    
    :param image: Original image
    :param regions: List of regions
    :param threshold: Similarity threshold for merging
    :return: Segmented image
    """
    segmented = np.zeros_like(image, dtype=int)
    label_counter = 1

    for region in regions:
        if region.size > 0:  # Only process non-empty regions
            y, x = np.unravel_index(np.argmin(np.abs(image - region[0, 0])), image.shape)
            h, w = region.shape
            segmented[y:y+h, x:x+w] = label_counter
            label_counter += 1

    changed = True
    while changed:
        changed = False
        props = regionprops(segmented)
        for prop in props:
            for neighbor in props:
                if prop.label != neighbor.label:
                    if abs(np.mean(image[segmented == prop.label]) - np.mean(image[segmented == neighbor.label])) <= threshold:
                        segmented[segmented == neighbor.label] = prop.label
                        changed = True
                        break
            if changed:
                break

    return segmented

def split_and_merge(image, split_threshold, merge_threshold):
    """
    Perform split-and-merge segmentation
    
    :param image: Input image (2D numpy array)
    :param split_threshold: Threshold for splitting
    :param merge_threshold: Threshold for merging
    :return: Segmented image
    """
    regions = split_image(image, split_threshold)
    segmented = merge_regions(image, regions, merge_threshold)
    
    return segmented

# Example usage:
image = cv2.imread('Kirat_King_(_Yalamber_)_Sankhuwasabha,_Nepal.jpg', 0)  # Read as grayscale
split_threshold = 20
merge_threshold = 10
result = split_and_merge(image, split_threshold, merge_threshold)

# Visualize the result
colored_result = label2rgb(result, image=image, bg_label=0)
cv2.imwrite('segmented.png', img_as_ubyte(colored_result))
