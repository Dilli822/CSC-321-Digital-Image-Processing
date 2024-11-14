import cv2
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class ImageRestoration:
    def __init__(self, image_path):
        self.img = cv2.imread(image_path, 0)
        if self.img is None:
            raise FileNotFoundError("Image not found")
    
    def add_noise(self, noise_type='gaussian', params=None):
        """Add different types of noise to image"""
        noisy = np.float32(self.img.copy())
        
        if noise_type == 'gaussian':
            mean, var = params or (0, 50)
            noise = np.random.normal(mean, var**0.5, self.img.shape)
            noisy += noise
            
        elif noise_type == 'salt_pepper':
            prob = params or 0.05
            mask = np.random.random(self.img.shape) < prob
            noisy[mask] = 255
            mask = np.random.random(self.img.shape) < prob
            noisy[mask] = 0
            
        return np.uint8(np.clip(noisy, 0, 255))

    def mean_filters(self, img, kernel_size=3, filter_type='arithmetic'):
        """Apply different mean filters"""
        if filter_type == 'arithmetic':
            return cv2.blur(img, (kernel_size, kernel_size))
            
        elif filter_type == 'geometric':
            kernel = np.ones((kernel_size, kernel_size))
            dst = cv2.filter2D(np.float32(img), -1, kernel)
            dst = np.exp(dst/(kernel_size**2))
            return np.uint8(np.clip(dst, 0, 255))
            
        elif filter_type == 'harmonic':
            kernel = np.ones((kernel_size, kernel_size))
            return cv2.filter2D(1.0/img, -1, kernel)
            
        return img

    def order_statistics_filters(self, img, kernel_size=3, filter_type='median'):
        """Apply order statistics filters"""
        if filter_type == 'median':
            return cv2.medianBlur(img, kernel_size)
            
        elif filter_type == 'min':
            return cv2.erode(img, np.ones((kernel_size, kernel_size)))
            
        elif filter_type == 'max':
            return cv2.dilate(img, np.ones((kernel_size, kernel_size)))
            
        return img

    def bandpass_filter(self, low_cut, high_cut, filter_type='ideal'):
        """Apply bandpass filter in frequency domain"""
        dft = cv2.dft(np.float32(self.img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        rows, cols = self.img.shape
        crow, ccol = rows//2, cols//2
        
        mask = np.zeros((rows, cols))
        for x in range(rows):
            for y in range(cols):
                d = np.sqrt((x-crow)**2 + (y-ccol)**2)
                
                if filter_type == 'ideal':
                    if low_cut <= d <= high_cut:
                        mask[x,y] = 1
                        
                elif filter_type == 'butterworth':
                    n = 2  # Order of filter
                    mask[x,y] = 1 / (1 + (d/low_cut)**(2*n)) * (1 - 1/(1 + (d/high_cut)**(2*n)))
                    
                elif filter_type == 'gaussian':
                    mask[x,y] = np.exp(-((d**2-low_cut**2)/(d*high_cut))**2)
        
        mask = np.float32(mask)
        fshift = dft_shift * mask[:,:,np.newaxis]
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
        
        return np.uint8(cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX))
def main():
    # Initialize
    restorer = ImageRestoration('Examples-of-raw-fluorescence-microscopy-images-and-their-estimated-ground-truth-from-our.png')
    
    # Add noise
    noisy_gaussian = restorer.add_noise('gaussian', (0, 50))
    noisy_sp = restorer.add_noise('salt_pepper', 0.05)
    
    # Apply mean filters
    mean_filtered = restorer.mean_filters(noisy_gaussian, 3, 'arithmetic')
    geometric_filtered = restorer.mean_filters(noisy_gaussian, 3, 'geometric')
    
    # Apply order statistics filters
    median_filtered = restorer.order_statistics_filters(noisy_sp, 3, 'median')
    min_filtered = restorer.order_statistics_filters(noisy_sp, 3, 'min')
    
    # Apply bandpass filter
    bandpass = restorer.bandpass_filter(30, 80, 'gaussian')
    
    # Prepare results for display
    results = {
        'Original': restorer.img,
        'Noisy (Gaussian)': noisy_gaussian,
        'Mean Filter': mean_filtered,
        'Median Filter': median_filtered,
        'Bandpass Filter': bandpass
    }
    
    # Plot images in a 2x3 grid layout
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('Image Restoration Results', fontsize=16)

    # List of titles and images
    titles = list(results.keys())
    images = list(results.values())
    
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i], cmap='gray')
            ax.set_title(titles[i])
        ax.axis('off')  # Hide axis

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

if __name__ == "__main__":
    main()