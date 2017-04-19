import matplotlib as mlp
mlp.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt

# In this example, we will make the graphs in the following link - 
# https://drive.google.com/file/d/0B0yTQeFvwuNjekg2eFdMZVhHYm8/view

# Random data generation
np.random.seed(1874)
x, y, z = np.random.normal(0, 1, (3, 100))
t = np.arctan2(y, x)
size = 50 * np.cos(2 * t)**2 + 10

fig, axes = plt.subplots(ncols = 3, figsize = plt.figaspect(0.5))

axes[0].scatter(x, y, marker = "o", s = 50)
axes[1].scatter(x, y, s = size, marker = "s", color = "darkblue")
axes[2].scatter(x, y, c = z, s = size, cmap = "gist_ncar")

for num in [0, 1, 2]:
    axes[num].set(xticks = [], yticks = [])

plt.show()
