def compute_signature(contour):
    moments = cv2.moments(contour)
    cx = int(moments['m10'] / moments['m00'])  # Centroid x
    cy = int(moments['m01'] / moments['m00'])  # Centroid y
    
    distances = []
    for point in contour:
        x, y = point[0]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        distances.append(dist)
    
    return distances

# Compute the signature for the first contour
signature = compute_signature(contours[0])

# Visualize the result
plt.plot(signature)
plt.title("Shape Signature")
plt.show()
