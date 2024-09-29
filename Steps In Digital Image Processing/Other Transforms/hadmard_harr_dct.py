import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
from pywt import wavedec
from scipy.fftpack import dct

def generate_sample_signal(n):
    """Generate a sample signal consisting of two sine waves."""
    t = np.linspace(0, 1, n, endpoint=False)
    return np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)

def hadamard_transform(signal):
    """Compute the Hadamard Transform of a signal."""
    n = len(signal)
    H = hadamard(n)  # Create Hadamard matrix of size n
    return np.dot(H, signal) / np.sqrt(n)  # Normalize the result

def haar_transform(signal):
    """Compute the Haar Transform of a signal."""
    coeffs = wavedec(signal, 'haar', level=int(np.log2(len(signal))))
    return np.concatenate(coeffs)  # Concatenate coefficients into a single array

def plot_transforms(signal, hadamard, haar, dct_coeffs):
    """Plot the original signal and its transforms."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Signal Transforms Visualization')

    axs[0, 0].plot(signal)
    axs[0, 0].set_title('Original Signal')
    axs[0, 0].set_xlabel('Sample')
    axs[0, 0].set_ylabel('Amplitude')

    axs[0, 1].plot(hadamard)
    axs[0, 1].set_title('Hadamard Transform')
    axs[0, 1].set_xlabel('Coefficient')
    axs[0, 1].set_ylabel('Magnitude')

    axs[1, 0].plot(haar)
    axs[1, 0].set_title('Haar Transform')
    axs[1, 0].set_xlabel('Coefficient')
    axs[1, 0].set_ylabel('Magnitude')

    axs[1, 1].plot(dct_coeffs)
    axs[1, 1].set_title('Discrete Cosine Transform')
    axs[1, 1].set_xlabel('Coefficient')
    axs[1, 1].set_ylabel('Magnitude')

    plt.tight_layout()
    plt.show()

# Generate sample signal
n = 256
signal = generate_sample_signal(n)

# Apply transforms
hadamard_coeffs = hadamard_transform(signal)
haar_coeffs = haar_transform(signal)
dct_coeffs = dct(signal, type=2, norm='ortho')

# Plot results
plot_transforms(signal, hadamard_coeffs, haar_coeffs, dct_coeffs)


import numpy as np
import matplotlib.pyplot as plt
import pywt

# Sample signal
Fs = 1000  # Sampling frequency
T = 1 / Fs  # Sampling interval
t = np.arange(0, 1, T)  # Time vector
signal = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)

# Pad signal to next power of 2 for Hadamard Transform
N = len(signal)
N_padded = 2 ** np.ceil(np.log2(N)).astype(int)
padded_signal = np.pad(signal, (0, N_padded - N), 'constant')

# Hadamard Transform
def hadamard_transform(signal):
    n = len(signal)
    hadamard_matrix = np.array([[1, 1], [1, -1]])
    
    # Construct Hadamard matrix using Kronecker product
    while hadamard_matrix.shape[0] < n:
        hadamard_matrix = np.kron(hadamard_matrix, [[1, 1], [1, -1]])
    
    return np.dot(hadamard_matrix, signal)

# Haar Transform
def haar_transform(signal):
    coeffs = pywt.wavedec(signal, 'haar')
    return np.concatenate(coeffs)

# Discrete Cosine Transform
def dct_transform(signal):
    return np.fft.fft(signal)[:len(signal) // 2]

# Perform the transformations
hadamard_values = hadamard_transform(padded_signal)
haar_coefficients = haar_transform(padded_signal)
dct_values = dct_transform(padded_signal)

# Plotting
fig, axs = plt.subplots(4, 1, figsize=(12, 12))

# Original Signal
axs[0].plot(t, signal)
axs[0].set_title('Original Signal')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Amplitude')
axs[0].grid()

# Hadamard Transform
axs[1].plot(hadamard_values)
axs[1].set_title('Hadamard Transform')
axs[1].set_xlabel('Coefficient Index')
axs[1].set_ylabel('Value')
axs[1].grid()

# Haar Transform
axs[2].plot(haar_coefficients)
axs[2].set_title('Haar Transform')
axs[2].set_xlabel('Coefficient Index')
axs[2].set_ylabel('Value')
axs[2].grid()

# DCT Transform
axs[3].plot(np.abs(dct_values))
axs[3].set_title('Discrete Cosine Transform (DCT)')
axs[3].set_xlabel('Frequency Index')
axs[3].set_ylabel('Magnitude')
axs[3].grid()

plt.tight_layout()
plt.show()
