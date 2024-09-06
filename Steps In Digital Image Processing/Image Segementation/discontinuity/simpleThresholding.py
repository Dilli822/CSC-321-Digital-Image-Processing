# import cv2
# import numpy as np

# def simple_thresholding(image_path, output_path, threshold_value):
#     # Load the image
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
#     # Apply binary thresholding
#     _, segmented = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    
#     # Save the result
#     cv2.imwrite(output_path, segmented)

# # Example usage
# image_path = 'yalamber.jpg'  # Path to your image file
# output_path = 'thresholded_result.png'
# threshold_value = 128  # Adjust this value based on your needs
# simple_thresholding(image_path, output_path, threshold_value)


# import cv2
# import numpy as np

# def grabcut_segmentation(image_path, output_path, rect):
#     # Load the image
#     image = cv2.imread(image_path)
#     mask = np.zeros(image.shape[:2], np.uint8)
    
#     # Create background and foreground models
#     bgd_model = np.zeros((1, 65), np.float64)
#     fgd_model = np.zeros((1, 65), np.float64)
    
#     # Apply GrabCut algorithm
#     cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
#     # Modify the mask to binary format
#     mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
#     segmented = image * mask2[:, :, np.newaxis]
    
#     # Save the result
#     cv2.imwrite(output_path, segmented)

# # Example usage
# image_path = 'yalamber.jpg'  # Path to your image file
# output_path = 'grabcut_result.png'
# rect = (50, 50, 450, 300)  # Define the rectangle around the object (x, y, width, height)
# grabcut_segmentation(image_path, output_path, rect)


import cv2
import numpy as np

def contour_detection(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply binary thresholding
    _, thresholded = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the original image
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(color_image, contours, -1, (0, 255, 0), 2)
    
    # Save the result
    cv2.imwrite(output_path, color_image)

# Example usage
image_path = 'Kirat_King_(_Yalamber_)_Sankhuwasabha,_Nepal.jpg'  # Path to your image file
output_path = 'Kcontours_result.png'
contour_detection(image_path, output_path)
