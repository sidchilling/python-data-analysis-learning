import matplotlib as mpl
mpl.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.cbook import get_sample_data as getSampleData
data = np.load(getSampleData("axes_grid/bivariate_normal.npy"))

print "Data: {}".format(data)
print "Data Shape: {}".format(data.shape)

fig, ax = plt.subplots()
im = ax.imshow(data, cmap = "gist_earth")
fig.colorbar(im) # Note: Colorbar is a `figure` method, not of `axis`

plt.show()

fig, ax = plt.subplots()
cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
im = ax.imshow(data, cmap = "gist_earth")
fig.colorbar(im, cax = cax, orientation = "horizontal")
plt.show()

fig, ax = plt.subplots()
im = ax.imshow(data, cmap = "seismic", interpolation = "nearest", vmin = -2, vmax = 2)
fig.colorbar(im)
plt.show()