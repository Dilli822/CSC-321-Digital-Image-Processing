import numpy as np
import matplotlib.pyplot as plt

# Create a blank grayscale image (all zeros)
image_shape = (256, 256)  # Define the size of the blank image
blank_image = np.zeros(image_shape, dtype=np.uint8)

# Function to add Gaussian noise
def add_gaussian_noise(image, mean=0, sigma=25):
    gaussian_noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + gaussian_noise.astype(np.float32)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

# Function to add Rayleigh noise
def add_rayleigh_noise(image, scale=25):
    rayleigh_noise = np.random.rayleigh(scale, image.shape)
    noisy_image = image + rayleigh_noise.astype(np.float32)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

# Function to add Uniform noise
def add_uniform_noise(image, low=-25, high=25):
    uniform_noise = np.random.uniform(low, high, image.shape)
    noisy_image = image + uniform_noise.astype(np.float32)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

# Function to add Erlang noise
def add_erlang_noise(image, k=2, theta=10):
    erlang_noise = np.random.gamma(k, theta, image.shape)
    noisy_image = image + erlang_noise.astype(np.float32)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

# Function to add Exponential noise
def add_exponential_noise(image, scale=25):
    exponential_noise = np.random.exponential(scale, image.shape)
    noisy_image = image + exponential_noise.astype(np.float32)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

# Function to add Impulsive (Salt and Pepper) noise
def add_impulsive_noise(image, prob=0.02):
    noisy_image = np.copy(image)
    total_pixels = image.size
    num_impulsive = int(prob * total_pixels)
    
    # Add salt (white)
    salt_coords = [np.random.randint(0, i, num_impulsive) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 255
    
    # Add pepper (black)
    pepper_coords = [np.random.randint(0, i, num_impulsive) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0
    
    return noisy_image

# Apply different types of noise to the blank image
noisy_gaussian = add_gaussian_noise(blank_image)
noisy_rayleigh = add_rayleigh_noise(blank_image)
noisy_uniform = add_uniform_noise(blank_image)
noisy_erlang = add_erlang_noise(blank_image)
noisy_exponential = add_exponential_noise(blank_image)
noisy_impulsive = add_impulsive_noise(blank_image)

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

plt.tight_layout()
plt.show()
