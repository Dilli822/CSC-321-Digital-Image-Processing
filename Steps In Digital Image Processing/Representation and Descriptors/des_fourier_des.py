def fourier_descriptors(contour):
    contour_complex = np.array([complex(p[0][0], p[0][1]) for p in contour])
    fourier_result = np.fft.fft(contour_complex)
    return fourier_result

# Compute Fourier Descriptors
fd = fourier_descriptors(contours[0])

# Visualize Fourier Descriptors
plt.plot(np.abs(fd))
plt.title("Fourier Descriptors")
plt.show()
