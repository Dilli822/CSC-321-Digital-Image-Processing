import requests
from PIL import Image
from io import BytesIO
import heapq
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Step 1: Download image from URL
def download_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img.convert('L')  # Convert to grayscale

# Step 2: Calculate the frequency of pixel values
def calculate_frequency(image):
    pixel_values = list(image.getdata())
    return Counter(pixel_values)

# Step 3: Build the Huffman tree
class HuffmanNode:
    def __init__(self, freq, pixel_value=None, left=None, right=None):
        self.freq = freq
        self.pixel_value = pixel_value
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(frequency):
    heap = []
    for pixel_value, freq in frequency.items():
        heapq.heappush(heap, HuffmanNode(freq, pixel_value))
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)
    
    return heap[0]

# Step 4: Generate Huffman codes
def generate_huffman_codes(node, code="", codes=None):
    if codes is None:
        codes = {}
    if node is None:
        return
    if node.pixel_value is not None:
        codes[node.pixel_value] = code
    generate_huffman_codes(node.left, code + "0", codes)
    generate_huffman_codes(node.right, code + "1", codes)
    return codes

# Step 5: Encode image using Huffman codes
def encode_image(image, huffman_codes):
    pixel_values = list(image.getdata())
    encoded_data = "".join([huffman_codes[pixel] for pixel in pixel_values])
    return encoded_data

# Step 6: Decode the encoded data using Huffman tree
def decode_image(encoded_data, huffman_tree, width, height):
    decoded_pixels = []
    node = huffman_tree
    for bit in encoded_data:
        if bit == '0':
            node = node.left
        else:
            node = node.right
        if node.pixel_value is not None:
            decoded_pixels.append(node.pixel_value)
            node = huffman_tree
    
    decoded_image = Image.new('L', (width, height))
    decoded_image.putdata(decoded_pixels)
    return decoded_image

# Function to calculate the size in bits
def calculate_image_size_in_bits(image):
    width, height = image.size
    return width * height * 8  # Each pixel is 8 bits in grayscale

# Function to calculate the size of the encoded data in bits
def calculate_encoded_size_in_bits(encoded_data):
    return len(encoded_data)  # Each '0' or '1' is 1 bit

# Function to plot the original and decompressed image matrices
def plot_image_matrix(image, title="Image Matrix"):
    plt.figure(figsize=(6, 6))
    plt.imshow(np.array(image), cmap='gray')
    plt.title(title)
    plt.colorbar()
    plt.show()

# Function to plot the image in 3D space
def plot_image_3d(image, title="3D Image Intensity Plot"):
    img_array = np.array(image)
    x, y = np.meshgrid(range(img_array.shape[1]), range(img_array.shape[0]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, img_array, cmap='gray')
    ax.set_title(title)
    plt.show()

# Main function to compress and decompress the image and print sizes
def huffman_compress_and_decompress_image(url):
    # Download and preprocess image
    original_image = download_image(url)
    width, height = original_image.size
    
    # Frequency calculation
    frequency = calculate_frequency(original_image)
    
    # Build Huffman Tree
    huffman_tree = build_huffman_tree(frequency)
    
    # Generate Huffman Codes
    huffman_codes = generate_huffman_codes(huffman_tree)
    
    # Encode image
    encoded_data = encode_image(original_image, huffman_codes)
    
    # Decode image
    decoded_image = decode_image(encoded_data, huffman_tree, width, height)
    
    # Show the original and decoded images
    original_image.show(title="Original Image")
    decoded_image.show(title="Decoded Image")

    # Print sizes and encoded data
    original_size_bits = calculate_image_size_in_bits(original_image)
    encoded_size_bits = calculate_encoded_size_in_bits(encoded_data)
    
    print(f"Original Image Size: {original_image.size} pixels")
    print(f"Original Image Size in Bits: {original_size_bits} bits")
    print(f"Encoded Data Size in Bits: {encoded_size_bits} bits")
    
    # Plotting the image matrix in 2D and 3D
    plot_image_matrix(original_image, title="Original Image Matrix")
    plot_image_matrix(decoded_image, title="Decoded Image Matrix")
    plot_image_3d(original_image, title="Original Image 3D Intensity")
    plot_image_3d(decoded_image, title="Decoded Image 3D Intensity")

# Example usage
# image_url = 'https://static.wixstatic.com/media/c712fa_daf9487042f644e8b052f3020ac14eac.jpg/v1/fill/w_1000,h_889,al_c,q_85,usm_0.66_1.00_0.01/c712fa_daf9487042f644e8b052f3020ac14eac.jpg'  
# image_url = 'https://mansanyak.wordpress.com/wp-content/uploads/2013/01/y_008.png?w=584'
# image_url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSYWsToWLjezgMch29YWFeGaD3LilWgxEUI7Q&s'
image_url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQWYom_4MIJofutZYwiUlqk3otVr2OV5fUznLmZwiEOOOvOUAU_AwJ2IcrUC6a9x_NT_pY&usqp=CAU'
huffman_compress_and_decompress_image(image_url)
