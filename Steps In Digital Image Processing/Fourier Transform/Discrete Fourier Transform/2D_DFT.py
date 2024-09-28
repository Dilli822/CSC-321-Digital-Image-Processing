import numpy as np
import matplotlib.pyplot as plt

# Create a 2D signal (e.g., an image)
M, N = 256, 256
x = np.linspace(-1, 1, M)
y = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, y)
Z = np.exp(-(X**2 + Y**2))  # Gaussian function as an example

# Perform 2D DFT
dft_values_2d = np.fft.fft2(Z)
dft_values_2d_shifted = np.fft.fftshift(dft_values_2d)  # Shift zero frequency to center
magnitude = np.abs(dft_values_2d_shifted)

# Plotting
plt.subplot(1, 2, 1)
plt.imshow(Z, cmap='gray')
plt.title("Original 2D Signal (Image)")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(np.log(1 + magnitude), cmap='gray')  # Log scale for visibility
plt.title("2D DFT Magnitude Spectrum")
plt.colorbar()

plt.show()
