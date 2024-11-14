import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_all_filters(img_path, cutoff=30):
    # Read image
    img = cv2.imread(img_path, 0)
    rows, cols = img.shape
    crow, ccol = rows//2, cols//2
    
    # Compute DFT
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Create coordinate grid
    x = np.arange(rows) - crow
    y = np.arange(cols) - ccol
    X, Y = np.meshgrid(y, x)
    D = np.sqrt(X**2 + Y**2)

    # Define all filters
    filters = {
        'Ideal LP': D <= cutoff,
        'Ideal HP': D > cutoff,
        'Butterworth LP': 1 / (1 + (D/cutoff)**4),
        'Butterworth HP': 1 / (1 + (cutoff/D)**4),
        'Gaussian LP': np.exp(-D**2/(2*cutoff**2)),
        'Gaussian HP': 1 - np.exp(-D**2/(2*cutoff**2)),
        'Laplacian': -4*np.pi**2 * (X**2 + Y**2)
    }

    # Apply each filter
    results = {}
    plt.figure(figsize=(20, 10))
    
    # Original image
    plt.subplot(2, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    # Apply and display each filter
    for i, (name, mask) in enumerate(filters.items(), 2):
        # Apply filter
        fshift = dft_shift * mask[:,:,np.newaxis]
        f_ishift = np.fft.ifftshift(fshift)
        filtered = cv2.idft(f_ishift)
        magnitude = cv2.magnitude(filtered[:,:,0], filtered[:,:,1])
        
        # Normalize result
        result = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        results[name] = result
        
        # Display
        plt.subplot(2, 4, i)
        plt.imshow(result, cmap='gray')
        plt.title(name)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    return results

# Apply filters
results = apply_all_filters('The-Results-Of-A-Head-CT-Scan.jpg', cutoff=30)