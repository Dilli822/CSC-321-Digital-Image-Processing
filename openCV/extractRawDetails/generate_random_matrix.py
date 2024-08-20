import numpy as np

def generate_random_matrix(rows, columns):
    return np.random.choice([0, 255], size=(rows, columns))

def save_matrix_to_txt(matrix, filename):
    np.savetxt(filename, matrix, fmt='%d', delimiter=' ')

# Example usage
rows, columns = 512, 256
random_matrix = generate_random_matrix(rows, columns)
save_matrix_to_txt(random_matrix, './random_matrix.txt')
