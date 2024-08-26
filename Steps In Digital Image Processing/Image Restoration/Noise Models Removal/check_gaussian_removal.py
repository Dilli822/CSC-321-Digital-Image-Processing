# # import numpy as np
# # import cv2
# # import matplotlib.pyplot as plt
# # from skimage.metrics import structural_similarity as ssim

# # def mse(imageA, imageB):
# #     """Calculate Mean Squared Error between two images."""
# #     return np.mean((imageA - imageB) ** 2)

# # def psnr(imageA, imageB):
# #     """Calculate Peak Signal-to-Noise Ratio between two images."""
# #     mse_val = mse(imageA, imageB)
# #     if mse_val == 0:
# #         return float('inf')
# #     return 20 * np.log10(255.0) - 10 * np.log10(mse_val)

# # def compare_images(original, noisy, denoised):
# #     """Compare original, noisy, and denoised images."""
# #     # Calculate MSE
# #     mse_noisy = mse(original, noisy)
# #     mse_denoised = mse(original, denoised)
    
# #     # Calculate PSNR
# #     psnr_noisy = psnr(original, noisy)
# #     psnr_denoised = psnr(original, denoised)
    
# #     # Calculate SSIM
# #     min_dim = min(original.shape[:2])
# #     win_size = min_dim if min_dim % 2 != 0 else min_dim - 1
# #     win_size = max(3, win_size)  # Ensure a minimum window size of 3x3
    
# #     if win_size > min_dim:
# #         print(f"Warning: Adjusted window size ({win_size}) to fit image dimensions.")
# #         win_size = min_dim  # Adjust window size to fit the image dimensions
    
# #     try:
# #         ssim_noisy = ssim(original, noisy, multichannel=True, win_size=win_size)
# #         ssim_denoised = ssim(original, denoised, multichannel=True, win_size=win_size)
# #     except ValueError as e:
# #         print(f"Error calculating SSIM: {e}")
# #         ssim_noisy = ssim_denoised = None
    
# #     print(f"MSE - Noisy: {mse_noisy:.2f}, Denoised: {mse_denoised:.2f}")
# #     print(f"PSNR - Noisy: {psnr_noisy:.2f}, Denoised: {psnr_denoised:.2f}")
# #     if ssim_noisy is not None and ssim_denoised is not None:
# #         print(f"SSIM - Noisy: {ssim_noisy:.4f}, Denoised: {ssim_denoised:.4f}")
    
# #     # Plot histograms
# #     plt.figure(figsize=(15,5))
# #     for i, img in enumerate([original, noisy, denoised]):
# #         plt.subplot(1, 3, i+1)
# #         plt.hist(img.ravel(), 256, [0,256])
# #         plt.title(['Original', 'Noisy', 'Denoised'][i])
# #     plt.tight_layout()
# #     plt.show()

# # # Load images
# # image_paths = {
# #     'original': '512x512-No-Noise.jpg',
# #     'noisy': '512x512-Gaussian-Noise.jpg',
# #     'denoised': 'denoised_image.jpg'
# # }

# # images = {}
# # for key, path in image_paths.items():
# #     image = cv2.imread(path)
# #     if image is None:
# #         print(f"Error: Could not read image at {path}")
# #     else:
# #         images[key] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # if len(images) != 3:
# #     raise ValueError("Could not read one or more images. Please check the file paths.")

# # # Extract images
# # original_rgb = images['original']
# # noisy_rgb = images['noisy']
# # denoised_rgb = images['denoised']

# # # Display results
# # plt.figure(figsize=(15,5))
# # for i, img in enumerate([original_rgb, noisy_rgb, denoised_rgb]):
# #     plt.subplot(1, 3, i+1)
# #     plt.imshow(img)
# #     plt.title(['Original', 'Noisy', 'Denoised'][i])
# #     plt.axis('off')
# # plt.tight_layout()
# # plt.show()

# # # Compare images
# # compare_images(original_rgb, noisy_rgb, denoised_rgb)


# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from skimage.metrics import structural_similarity as ssim
# from skimage.restoration import denoise_wavelet

# def add_gaussian_noise(image, mean=0, sigma=25):
#     """Add Gaussian noise to an image."""
#     row, col, ch = image.shape
#     gauss = np.random.normal(mean, sigma, (row, col, ch))
#     noisy = image + gauss
#     return np.clip(noisy, 0, 255).astype(np.uint8)

# def mse(imageA, imageB):
#     """Calculate Mean Squared Error between two images."""
#     return np.mean((imageA.astype("float") - imageB.astype("float")) ** 2)

# def psnr(imageA, imageB):
#     """Calculate Peak Signal-to-Noise Ratio between two images."""
#     mse_val = mse(imageA, imageB)
#     if mse_val == 0:
#         return float('inf')
#     return 20 * np.log10(255.0) - 10 * np.log10(mse_val)

# def image_stats(image, name):
#     """Print basic statistics of an image."""
#     print(f"{name} - Min: {np.min(image)}, Max: {np.max(image)}, Mean: {np.mean(image):.2f}, Std: {np.std(image):.2f}")

# def compare_images(original, noisy, denoised):
#     """Compare original, noisy, and denoised images."""
#     # Calculate MSE
#     mse_noisy = mse(original, noisy)
#     mse_denoised = mse(original, denoised)
    
#     # Calculate PSNR
#     psnr_noisy = psnr(original, noisy)
#     psnr_denoised = psnr(original, denoised)
    
#     # Calculate SSIM
#     ssim_noisy = ssim(original, noisy, multichannel=True)
#     ssim_denoised = ssim(original, denoised, multichannel=True)
    
#     print(f"MSE - Noisy: {mse_noisy:.2f}, Denoised: {mse_denoised:.2f}")
#     print(f"PSNR - Noisy: {psnr_noisy:.2f}, Denoised: {psnr_denoised:.2f}")
#     print(f"SSIM - Noisy: {ssim_noisy:.4f}, Denoised: {ssim_denoised:.4f}")

#     # Plot histograms
#     plt.figure(figsize=(15,5))
#     for i, img in enumerate([original, noisy, denoised]):
#         plt.subplot(1, 3, i+1)
#         plt.hist(img.ravel(), 256, [0,256])
#         plt.title(['Original', 'Noisy', 'Denoised'][i])
#     plt.tight_layout()
#     plt.show()

# # Load image
# image_path = '512x512-No-Noise.jpg'
# original = cv2.imread(image_path)
# if original is None:
#     raise ValueError(f"Could not read image at {image_path}")
# original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

# # Print original image stats
# image_stats(original, "Original")

# # Add Gaussian noise
# noisy = add_gaussian_noise(original, mean=0, sigma=25)
# image_stats(noisy, "Noisy")

# # Denoise using wavelet denoising
# denoised = denoise_wavelet(noisy, multichannel=True, convert2ycbcr=True, 
#                            method='BayesShrink', mode='soft', rescale_sigma=True)
# denoised = (denoised * 255).astype(np.uint8)
# image_stats(denoised, "Denoised")

# # Display results
# plt.figure(figsize=(15,5))
# for i, img in enumerate([original, noisy, denoised]):
#     plt.subplot(1, 3, i+1)
#     plt.imshow(img)
#     plt.title(['Original', 'Noisy', 'Denoised'][i])
#     plt.axis('off')
# plt.tight_layout()
# plt.show()

# # Compare images
# compare_images(original, noisy, denoised)

# # Save processed images
# cv2.imwrite('noisy_image.jpg', cv2.cvtColor(noisy, cv2.COLOR_RGB2BGR))
# cv2.imwrite('denoised_image.jpg', cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR))