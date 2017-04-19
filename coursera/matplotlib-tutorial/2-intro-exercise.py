import matplotlib as mpl
mpl.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt

# Create one figure with 3 subplots -- one below the other of cos functions

# Data
x = np.linspace(0, 10, 1000)
ys = [np.cos(x), np.cos(x + 1), np.cos(x + 2)]
names = ["Signal #1", "Signal #2", "Signal #3"]

# Solution
fig, axes = plt.subplots(nrows = len(ys), ncols = 1)
for rowNum in range(0, len(ys)):
    ax = axes[rowNum]
    ax.set(title = names[rowNum], xticks = [], yticks = [], xlim = [np.min(x), np.max(x)],
            ylim = [-1.1, 1.1])
    ax.plot(x, ys[rowNum], color = "black")
plt.show()
