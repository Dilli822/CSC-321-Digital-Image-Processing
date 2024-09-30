import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform, rayleigh, gamma, expon

# Set up figure and subplots (2 rows, 3 columns)
fig, axs = plt.subplots(2, 3, figsize=(10, 6))

# Gaussian (Normal) distribution
x_gaussian = np.linspace(-5, 5, 1000)
gaussian_pdf = norm.pdf(x_gaussian, 0, 1)
axs[0, 0].plot(x_gaussian, gaussian_pdf, label='Gaussian')
axs[0, 0].set_title('Gaussian')
axs[0, 0].set_ylim(0, 0.5)
axs[0, 0].grid(True)

# Impulse (Salt-and-Pepper) noise - Simple example for visualization
x_impulse = np.array([0, 0.5, 1])
y_impulse = np.array([1, 0, 1])
axs[0, 1].step(x_impulse, y_impulse, label='Impulse', where='mid')
axs[0, 1].set_title('Impulse (Salt-and-Pepper)')
axs[0, 1].set_ylim(0, 1.5)
axs[0, 1].grid(True)

# Uniform distribution
x_uniform = np.linspace(-1, 1, 1000)
uniform_pdf = uniform.pdf(x_uniform, -1, 2)
axs[0, 2].plot(x_uniform, uniform_pdf, label='Uniform')
axs[0, 2].set_title('Uniform')
axs[0, 2].set_ylim(0, 1.5)
axs[0, 2].grid(True)

# Rayleigh distribution
x_rayleigh = np.linspace(0, 5, 1000)
rayleigh_pdf = rayleigh.pdf(x_rayleigh)
axs[1, 0].plot(x_rayleigh, rayleigh_pdf, label='Rayleigh')
axs[1, 0].set_title('Rayleigh')
axs[1, 0].set_ylim(0, 1)
axs[1, 0].grid(True)

# Gamma (Erlang) distribution
x_gamma = np.linspace(0, 10, 1000)
gamma_pdf = gamma.pdf(x_gamma, a=2, scale=2)
axs[1, 1].plot(x_gamma, gamma_pdf, label='Gamma')
axs[1, 1].set_title('Gamma (Erlang)')
axs[1, 1].set_ylim(0, 0.4)
axs[1, 1].grid(True)

# Exponential distribution
x_exponential = np.linspace(0, 5, 1000)
exponential_pdf = expon.pdf(x_exponential, scale=1)
axs[1, 2].plot(x_exponential, exponential_pdf, label='Exponential')
axs[1, 2].set_title('Exponential')
axs[1, 2].set_ylim(0, 1)
axs[1, 2].grid(True)

# Adjust layout and show plot
plt.tight_layout()
plt.show()
