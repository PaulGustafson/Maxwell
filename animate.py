import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

frames = 150
increment = 10
timesteps = increment*frames
nx, ny = 256, 256

def load_ez(step):
    # Load the data from the file saved for each time step
    # Example of loading from text files, adjust based on your method of saving
    return np.loadtxt(f'build/data/ez_step_{step}.txt').reshape((nx, ny))

# Prepare for animation
fig, ax = plt.subplots()
cax = ax.imshow(load_ez(0), vmin=-0.1, vmax=0.1, cmap=sns.cm.rocket)  # Adjust vmin and vmax for your data range
fig.colorbar(cax)

def update(frame):
    ez_data = load_ez(frame*increment)
    cax.set_array(ez_data)
    return cax,

ani = animation.FuncAnimation(fig, update, frames=range(frames), interval=200)
ani.save('ez_animation.mp4', writer='ffmpeg', fps=5)  # For MP4
