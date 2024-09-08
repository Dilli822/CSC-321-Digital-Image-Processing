import numpy as np
import matplotlib.pyplot as plt

def hough_transform(points, theta_res=1, rho_res=1):
    # Convert theta from degrees to radians
    thetas = np.deg2rad(np.arange(-90.0, 90.0, theta_res))
    
    # Calculate max possible distance in the image
    x_max, y_max = np.max(points, axis=0)
    max_dist = np.ceil(np.sqrt(x_max**2 + y_max**2))
    
    # Create rho range
    rhos = np.arange(-max_dist, max_dist, rho_res)
    
    # Initialize the Hough accumulator
    accumulator = np.zeros((len(rhos), len(thetas)))
    
    # Precompute cosine and sine values
    cos_theta = np.cos(thetas)
    sin_theta = np.sin(thetas)
    
    # Perform the Hough Transform
    for x, y in points:
        for theta_idx in range(len(thetas)):
            rho = x * cos_theta[theta_idx] + y * sin_theta[theta_idx]
            rho_idx = np.argmin(np.abs(rhos - rho))
            accumulator[rho_idx, theta_idx] += 1
    
    return accumulator, thetas, rhos

# Generate sample points for two lines
np.random.seed(42)
line1 = np.array([(t, 2*t + 1) for t in range(100)]) + np.random.normal(0, 2, (100, 2))
line2 = np.array([(t, -0.5*t + 80) for t in range(100)]) + np.random.normal(0, 2, (100, 2))
points = np.vstack((line1, line2))

# Apply Hough Transform
accumulator, thetas, rhos = hough_transform(points)

# Plot original points
plt.figure(figsize=(18, 6))
plt.subplot(131)
plt.scatter(points[:, 0], points[:, 1], alpha=0.5)
plt.title('Original Points')
plt.xlabel('X')
plt.ylabel('Y')

# Plot Hough Transform result
plt.subplot(132)
plt.imshow(accumulator, extent=[np.rad2deg(thetas[0]), np.rad2deg(thetas[-1]), rhos[-1], rhos[0]], 
           aspect='auto', cmap='jet')
plt.title('Hough Transform')
plt.xlabel('Theta (degrees)')
plt.ylabel('Rho')
plt.colorbar(label='Votes')

# Find peaks in the Hough Transform
threshold = 0.6 * np.max(accumulator)
peaks = np.argwhere(accumulator > threshold)

# Plot detected lines
plt.subplot(133)
plt.scatter(points[:, 0], points[:, 1], alpha=0.5)
for rho_idx, theta_idx in peaks:
    rho = rhos[rho_idx]
    theta = thetas[theta_idx]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    plt.plot([x1, x2], [y1, y2], 'r-')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.title('Detected Lines')
plt.xlabel('X')
plt.ylabel('Y')

plt.tight_layout()
plt.show()
