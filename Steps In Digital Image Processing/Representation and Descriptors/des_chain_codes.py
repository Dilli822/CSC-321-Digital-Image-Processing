def get_chain_code(contour):
    chain_codes = []
    for i in range(1, len(contour)):
        diff = contour[i][0] - contour[i-1][0]
        if (diff == [1, 0]).all():
            chain_codes.append(0)  # Right
        elif (diff == [1, -1]).all():
            chain_codes.append(1)  # Top-Right
        elif (diff == [0, -1]).all():
            chain_codes.append(2)  # Top
        elif (diff == [-1, -1]).all():
            chain_codes.append(3)  # Top-Left
        elif (diff == [-1, 0]).all():
            chain_codes.append(4)  # Left
        elif (diff == [-1, 1]).all():
            chain_codes.append(5)  # Bottom-Left
        elif (diff == [0, 1]).all():
            chain_codes.append(6)  # Bottom
        elif (diff == [1, 1]).all():
            chain_codes.append(7)  # Bottom-Right
    return chain_codes

# Find contours of the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Apply chain code on the first contour
chain_code = get_chain_code(contours[0])

# Visualize the result
print("Chain Code:", chain_code)
plt.plot(chain_code)
plt.title("Chain Code Representation")
plt.show()
