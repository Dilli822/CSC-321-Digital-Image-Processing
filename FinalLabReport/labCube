import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
from itertools import product, combinations  # Fix: Importing necessary modules

# Define the figure and axis for the plot
fig = plt.figure(figsize=(10, 10), dpi=100)
ax = plt.axes(projection="3d")

# Grid parameters
n_cubes = 10  # Number of cubes along one axis
cube_size = 1  # Size of each cube
grid_range = np.linspace(-10, 10, n_cubes)

# Function to draw a cube in 3D space
def draw_cube(x, y, z):
    r = [0, cube_size]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s - e)) == r[1] - r[0]:
            ax.plot3D(*zip(s + [x, y, z], e + [x, y, z]), color="cyan")

# Function to generate frames for the animation
def update(frame):
    ax.cla()  # Clear previous frame
    ax.set_box_aspect([1,1,1])  # Keep the aspect ratio

    # Set limits and view
    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])
    ax.set_zlim([-15, 15])
    ax.view_init(30, frame)  # Rotate view for animation

    # Draw the grid of cubes
    for x in grid_range:
        for y in grid_range:
            for z in grid_range:
                draw_cube(x, y, z)

# Create animation
anim = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50)

# Save the animation as a gif in the current directory
anim.save("cube_grid_animation.gif", writer='pillow', fps=24)

# Show the plot (optional)
plt.show()
