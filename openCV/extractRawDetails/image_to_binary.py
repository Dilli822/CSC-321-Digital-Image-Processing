import cv2
import numpy as np

# Read the image from file
image_path = './what.jpg'  # Replace with your image file path
image = cv2.imread(image_path)

if image is None:
    print("Error: Unable to load the image. Please check the file path.")
else:
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    # Threshold value: 127 (can be adjusted)
    # Max value: 255 (for white)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    # Save the binary image
    cv2.imwrite('binary_image.jpg', binary_image)

    # Display the binary image
    # cv2.imshow('Binary Image', binary_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Print the binary image matrix
    print("Binary Image Matrix:")
    print(binary_image)
    rows, cols = binary_image.shape
    for i in range(rows):
        for j in range(cols):
            print(f"{binary_image[i, j]:3}", end=" ")
        print()  # Newline after each row
        
    for i in range(rows):
        # Print each row with values separated by spaces
        print(" ".join(f"{binary_image[i, j]:3}" for j in range(cols)))

    # Print and save the binary image matrix in a formatted way
    output_file_path = 'binary_image_matrix.txt'
    
    with open(output_file_path, 'w') as file:
        file.write("Binary Image Matrix:\n")
        rows, cols = binary_image.shape
        for i in range(rows):
            # Format each row as a string and write it to the file
            formatted_row = " ".join(f"{binary_image[i, j]:3}" for j in range(cols))
            file.write(f"{formatted_row}\n")

    print(f"Binary image matrix has been saved to {output_file_path}")