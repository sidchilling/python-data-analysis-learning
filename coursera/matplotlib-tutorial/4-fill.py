import matplotlib as mlp
mlp.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
y = np.random.randn(100).cumsum()
x = np.linspace(0, 10, 100)

print "X: {}".format(x)
print "Y: {}".format(y)

fig, axes = plt.subplots(ncols = 2)

## Fill the area bounded by a curve
axes[0].fill_between(x, y, color = "lightblue")
axes[0].set(xticks = [], yticks = [])

## Different colors for the areas above (positive and negative)
axes[1].fill_between(x, y, where = [True if yval > 0 else False for yval in y], color = "lightblue")
axes[1].fill_between(x, y, where = [False if yval > 0 else True for yval in y], color = "salmon")
axes[1].set(xticks = [], yticks = [])

plt.show()

x = np.linspace(0, 10, 200)
y1 = 2 * x + 1
y2 = 3 * x + 1.2
yMean = 0.5 * x * np.cos(2 * x) + 2.5 * x + 1.1

fig, ax = plt.subplots()

# Fill between `y1` and `y2`
ax.fill_between(x, y1, y2, color = "yellow")
# Plot the centerline
ax.plot(x, yMean, color = "black")
ax.set(xticks = [], yticks = [])

plt.show()


