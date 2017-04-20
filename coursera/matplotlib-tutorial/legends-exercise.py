import matplotlib as mpl
mpl.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np

# In this exercise, we will reproduce the following figure - 
# https://drive.google.com/file/d/0B0yTQeFvwuNjWTBUeS0yZnZZVDA/view

## Data Generation Provided

t = np.linspace(0, 2 * np.pi, 150)
x1, y1 = np.cos(t), np.sin(t)
x2, y2 = 2 * x1, 2 * y1

colors = ["darkred", "darkgreen"]

# Try to make two circles, scale the axes as shown, and add a legend

fig, ax = plt.subplots(nrows = 1, ncols = 1)
ax.plot(x1, y1, color = colors[0], label = "Inner")
ax.plot(x2, y2, color = colors[1], label = "Outer")
ax.legend(loc = "best")
ax.axis("equal")
ax.margins(0.05)
ax.set(xticks = [-2, -1, 0, 1, 2], yticks = [-2, -1, 0, 1, 2])

plt.show()