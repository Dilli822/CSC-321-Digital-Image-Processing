import numpy as np

def create_digit_one_matrix(rows, columns):
    matrix = np.zeros((rows, columns), dtype=int)

    # Define the digit "1" by setting specific columns to 255
    for i in range(rows):
        matrix[i, columns // 2] = 255  # Vertical line in the middle
    
    # Optional: Add a base for the digit "1"
    matrix[rows-2:, columns//2 - 1:columns//2 + 2] = 255

    return matrix

def save_matrix_to_txt(matrix, filename):
    np.savetxt(filename, matrix, fmt='%d', delimiter=' ')

# Example usage
rows, columns = 512,256
digit_one_matrix = create_digit_one_matrix(rows, columns)
save_matrix_to_txt(digit_one_matrix, 'digit_one_matrix.txt')

print(digit_one_matrix)
