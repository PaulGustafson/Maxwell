import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

import os
import glob

# Get the list of files in the data directory
data_dir = '/home/ubuntu/maxwells/Maxwell/build/data/'
files = glob.glob(os.path.join(data_dir, 'u_step_*.txt'))

# Calculate frames and increment
# Sort the files based on their step number
sorted_files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

# Calculate the total number of frames
frames = len(sorted_files)

# Calculate the increment
if frames > 1:
    first_step = int(sorted_files[0].split('_')[-1].split('.')[0])
    last_step = int(sorted_files[-1].split('_')[-1].split('.')[0])
    increment = (last_step - first_step) // (frames - 1)
else:
    increment = 1

timesteps = increment * frames

# Get nx and ny from the first file
with open(files[0], 'r') as f:
    first_line = f.readline().strip().split()
    ny = 1
    nx = len(first_line)
    for line in f:
        ny += 1

print(f"Detected parameters: frames={frames}, increment={increment}, timesteps={timesteps}, nx={nx}, ny={ny}")

def load_ez(step):
    # Load the data from the file saved for each time step
    # Example of loading from text files, adjust based on your method of saving
    filename = f'/home/ubuntu/maxwells/Maxwell/build/data/u_step_{step}.txt'
    print("loading", filename)
    return np.loadtxt(filename).reshape((nx, ny))

# Prepare for animation
fig, ax = plt.subplots()
cax = ax.imshow(load_ez(0), vmin=-0.1, vmax=0.1, cmap=sns.cm.rocket)  # Adjust vmin and vmax for your data range
fig.colorbar(cax)

def update(frame):
    ez_data = load_ez(frame*increment)
    cax.set_array(ez_data)
    return cax

ani = animation.FuncAnimation(fig, update, frames=range(frames), interval=increment)
ani.save('ez_animation.mp4', writer='ffmpeg', fps=10)  # For MP4
