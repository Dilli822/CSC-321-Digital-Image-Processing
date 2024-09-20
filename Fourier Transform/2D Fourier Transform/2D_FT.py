import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlopen
from PIL import Image
import io

def load_and_resize_image_from_url(url, size=(150, 150)):
    try:
        with urlopen(url) as response:
            image_data = response.read()
        img = Image.open(io.BytesIO(image_data)).convert('L')  # Convert to grayscale
        img = img.resize(size, Image.Resampling.LANCZOS)  # Resize image with LANCZOS resampling
        return np.array(img)
    except Exception as e:
        print(f"Error loading or processing image: {e}")
        return None

def manual_2d_fourier_transform(image):
    M, N = image.shape
    F = np.zeros((M, N), dtype=complex)
    
    x = np.arange(M).reshape(-1, 1)
    y = np.arange(N)
    
    for u in range(M):
        for v in range(N):
            e = np.exp(-2j * np.pi * ((u * x / M) + (v * y / N)))
            F[u, v] = np.sum(image * e)
    
    return F

def magnitude_spectrum(F):
    return np.fft.fftshift(np.abs(F))

def visualize_results(image, magnitude_spectrum):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(np.log1p(magnitude_spectrum), cmap='viridis')
    plt.title("Magnitude Spectrum (Log Scale)")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

def main():
    # image_url = 'https://upload.wikimedia.org/wikipedia/commons/4/4f/Black_hole_-_Messier_87_crop_max_res.jpg'  # Replace with a valid image URL
    image_url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQiqlZ8DmOfJkyRnU4GyxllYRG8RePSqZlTxw&s'
    image = load_and_resize_image_from_url(image_url)
    if image is not None:
        F = manual_2d_fourier_transform(image)
        magnitude = magnitude_spectrum(F)
        visualize_results(image, magnitude)
    else:
        print("Failed to load image.")

if __name__ == "__main__":
    main()
