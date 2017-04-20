import matplotlib as mpl
mpl.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np

# In this exercise, we will reproduce the following figure -
# https://drive.google.com/file/d/0B0yTQeFvwuNjb3VVcjNVMHRzUk0/view

## Data Generation Provided

np.random.seed(1)

# Generate random data with different ranges...
data1 = np.random.random((10, 10))
data2 = 2 * np.random.random((10, 10))
data3 = 3 * np.random.random((10, 10))

print "Data1: {}".format(data1)
print "Data2: {}".format(data2)
print "Data3: {}".format(data3)

print "Data1 Shape: {}".format(data1.shape)
print "Data2 Shape: {}".format(data2.shape)
print "Data3 Shape: {}".format(data3.shape)

# Setup our figure and axes
fig, axes = plt.subplots(ncols = 3, figsize = plt.figaspect(0.5))
fig.tight_layout() # Make the subplots fill the figure a bit more
cax = fig.add_axes([0.25, 0.1, 0.55, 0.03]) # Add an axes for the colorbar

# Now, you're on your own

for ax, data in zip(axes, [data1, data2, data3]):
    im = ax.imshow(data, vmin = 0, vmax = 3, interpolation = "nearest")

fig.colorbar(im, cax = cax, orientation = "horizontal")
plt.show()