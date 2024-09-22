import numpy as np

def chain_code(binary_image):
    rows, cols = np.where(binary_image == 1)
    directions = []
    for i in range(1, len(rows)):
        dy = rows[i] - rows[i - 1]
        dx = cols[i] - cols[i - 1]
        
        if dy == 0 and dx == 1:
            directions.append(0)  # Right
        elif dy == -1 and dx == 1:
            directions.append(1)  # Top-Right
        elif dy == -1 and dx == 0:
            directions.append(2)  # Top
        elif dy == -1 and dx == -1:
            directions.append(3)  # Top-Left
        elif dy == 0 and dx == -1:
            directions.append(4)  # Left
        elif dy == 1 and dx == -1:
            directions.append(5)  # Bottom-Left
        elif dy == 1 and dx == 0:
            directions.append(6)  # Bottom
        elif dy == 1 and dx == 1:
            directions.append(7)  # Bottom-Right
    
    return directions

# Example usage with a binary image
binary_image = np.array([[0, 1, 0],
                         [1, 1, 0],
                         [0, 1, 0]])
chain_codes = chain_code(binary_image)
print(chain_codes)


# Signatures
import numpy as np

def compute_signature(binary_image):
    rows, cols = np.where(binary_image == 1)
    centroid_x = np.mean(cols)
    centroid_y = np.mean(rows)
    
    signature = []
    for i in range(len(rows)):
        distance = np.sqrt((cols[i] - centroid_x) ** 2 + (rows[i] - centroid_y) ** 2)
        signature.append(distance)
    
    return signature

# Example usage
binary_image = np.array([[0, 1, 0],
                         [1, 1, 0],
                         [0, 1, 0]])
signature = compute_signature(binary_image)
print(signature)


# shape numbers
def shape_number(chain_code):
    n = len(chain_code)
    rotations = [chain_code[i:] + chain_code[:i] for i in range(n)]
    min_rotation = min(rotations)
    return min_rotation

# Example usage with a chain code
chain_code_seq = [0, 1, 2, 3, 4, 5, 6, 7]
shape_num = shape_number(chain_code_seq)
print(shape_num)

# Fourier descriptors
import numpy as np

def fourier_descriptors(binary_image):
    rows, cols = np.where(binary_image == 1)
    boundary_points = cols + 1j * rows  # Represent boundary points as complex numbers
    fourier_desc = np.fft.fft(boundary_points)
    
    return fourier_desc

# Example usage
binary_image = np.array([[0, 1, 0],
                         [1, 1, 0],
                         [0, 1, 0]])
fd = fourier_descriptors(binary_image)
print(fd)
