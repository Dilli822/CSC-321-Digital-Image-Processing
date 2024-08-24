import cv2
import numpy as np

def ideal_high_pass_filter(image, cutoff):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    mask = np.ones((rows, cols), np.float32)
    for x in range(rows):
        for y in range(cols):
            if np.sqrt((x - crow)**2 + (y - ccol)**2) <= cutoff:
                mask[x, y] = 0
    
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    return img_back

def butterworth_high_pass_filter(image, cutoff, order):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    mask = np.zeros((rows, cols), np.float32)
    for x in range(rows):
        for y in range(cols):
            d = np.sqrt((x - crow)**2 + (y - ccol)**2)
            mask[x, y] = 1 / (1 + (cutoff / d)**(2*order))
    
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    return img_back

def gaussian_high_pass_filter(image, cutoff):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    mask = np.zeros((rows, cols), np.float32)
    for x in range(rows):
        for y in range(cols):
            d = np.sqrt((x - crow)**2 + (y - ccol)**2)
            mask[x, y] = 1 - np.exp(-(d**2) / (2 * cutoff**2))
    
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    return img_back

# Example usage
image = cv2.imread('./naruto.jpeg', 0)  # Read image in grayscale
cutoff = 30  # Adjust this value as needed
order = 2  # For Butterworth filter

ideal_filtered = ideal_high_pass_filter(image, cutoff)
butterworth_filtered = butterworth_high_pass_filter(image, cutoff, order)
gaussian_filtered = gaussian_high_pass_filter(image, cutoff)

# Normalize the filtered images for better visualization
ideal_filtered = cv2.normalize(ideal_filtered, None, 0, 255, cv2.NORM_MINMAX)
butterworth_filtered = cv2.normalize(butterworth_filtered, None, 0, 255, cv2.NORM_MINMAX)
gaussian_filtered = cv2.normalize(gaussian_filtered, None, 0, 255, cv2.NORM_MINMAX)

cv2.imshow('Original', image)
cv2.imshow('Ideal High-Pass Filter', ideal_filtered.astype(np.uint8))
cv2.imshow('Butterworth High-Pass Filter', butterworth_filtered.astype(np.uint8))
cv2.imshow('Gaussian High-Pass Filter', gaussian_filtered.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()