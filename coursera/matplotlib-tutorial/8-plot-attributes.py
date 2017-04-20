import matplotlib as mpl
mpl.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(ncols = 2, figsize = plt.figaspect(0.5))

t = np.arange(0.0, 5.0, 0.2)
# Red dashes, Blue Squares, and Green Triangles
axes[0].plot(t, t, "r--", t, t ** 2, "bs", t, t ** 3, "g^")

# Dotted Red line, with large yellow diamond markers that have a green edge
t = np.arange(0.0, 5.0, 0.1)
a = np.exp(-t) * np.cos(2 * np.pi * t)
axes[1].plot(t, a, color = "red", linestyle = ":", marker = "D", markerfacecolor = "yellow",
         markersize = 6, markeredgecolor = "green")
plt.show()