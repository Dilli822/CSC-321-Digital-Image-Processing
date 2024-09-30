import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import numpy as np

# Load image directly from URL
def load_image_from_url(url):
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Error: Cannot load image from '{url}'.")
        exit()
    
    return image

# Image URL (replace with your own URL)
image_url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSnPyRbNaOeQ-uFQqU_ek_HmLUIEVcxNAZ2SA&s'  # Example image URL

# Load image from the URL
image = load_image_from_url(image_url)

# Function to add Gaussian noise
def add_gaussian_noise(image, mean=0, sigma=25):
    gaussian_noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = cv2.add(image, gaussian_noise.astype(np.uint8))
    return noisy_image

# Function to add Rayleigh noise
def add_rayleigh_noise(image, scale=25):
    rayleigh_noise = np.random.rayleigh(scale, image.shape)
    noisy_image = cv2.add(image, rayleigh_noise.astype(np.uint8))
    return noisy_image

# Function to add Uniform noise
def add_uniform_noise(image, low=-25, high=25):
    uniform_noise = np.random.uniform(low, high, image.shape)
    noisy_image = cv2.add(image, uniform_noise.astype(np.uint8))
    return noisy_image

# Function to add Erlang noise
def add_erlang_noise(image, k=2, theta=10):
    erlang_noise = np.random.gamma(k, theta, image.shape)
    noisy_image = cv2.add(image, erlang_noise.astype(np.uint8))
    return noisy_image

# Function to add Exponential noise
def add_exponential_noise(image, scale=25):
    exponential_noise = np.random.exponential(scale, image.shape)
    noisy_image = cv2.add(image, exponential_noise.astype(np.uint8))
    return noisy_image

# Function to add Impulsive (Salt and Pepper) noise
def add_impulsive_noise(image, prob=0.02):
    noisy_image = np.copy(image)
    total_pixels = image.size
    num_impulsive = np.ceil(prob * total_pixels)
    
    # Add salt (white)
    coords = [np.random.randint(0, i - 1, int(num_impulsive)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 255  # Salt (white)
    
    # Add pepper (black)
    coords = [np.random.randint(0, i - 1, int(num_impulsive)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0  # Pepper (black)
    
    return noisy_image

# Apply different types of noise to the image
noisy_gaussian = add_gaussian_noise(image)
noisy_rayleigh = add_rayleigh_noise(image)
noisy_uniform = add_uniform_noise(image)
noisy_erlang = add_erlang_noise(image)
noisy_exponential = add_exponential_noise(image)
noisy_impulsive = add_impulsive_noise(image)

# Display all the noisy images
plt.figure(figsize=(12, 10))

plt.subplot(2, 3, 1)
plt.title('Gaussian Noise')
plt.imshow(noisy_gaussian, cmap='gray')

plt.subplot(2, 3, 2)
plt.title('Rayleigh Noise')
plt.imshow(noisy_rayleigh, cmap='gray')

plt.subplot(2, 3, 3)
plt.title('Uniform Noise')
plt.imshow(noisy_uniform, cmap='gray')

plt.subplot(2, 3, 4)
plt.title('Erlang Noise')
plt.imshow(noisy_erlang, cmap='gray')

plt.subplot(2, 3, 5)
plt.title('Exponential Noise')
plt.imshow(noisy_exponential, cmap='gray')

plt.subplot(2, 3, 6)
plt.title('Impulsive Noise')
plt.imshow(noisy_impulsive, cmap='gray')

plt.show()
