import matplotlib as mpl
mpl.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np

# By default, matplotlib will determine axes limits from our data. For line and image plots, 
# no extra padding is is added, while bar and scatter plots are given some padding.

fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = plt.figaspect(0.5))
axes[0].plot([-10, -5, 0, 5, 10, 15], [-1.2, 2, 3.5, -0.3, -4, 1])
axes[1].scatter([-10, -5, 0, 5, 10, 15], [-1.2, 2, 3.5, -0.3, -4, 1])

# We can add "a little bit of padding"
axes[0].margins(x = 0.0, y = 0.1) # 10% padding on the y-axis
axes[1].margins(0.05) # 5% padding on all directions

plt.show()

# Using `ax.axis()` method, we can set the limits on axis. But normally, we use the method
# to make the axes `equal` or `tight`.

fig, axes = plt.subplots(nrows = 3)
for ax in axes:
    ax.plot([-10, -5, 0, 5, 10, 15], [-1.2, 2, 3.5, -0.3, -4, 1])

axes[0].set_title("Normal Autoscaling", y = 0.7, x = 0.8)

axes[1].set_title("ax.axis('tight')", y = 0.7, x = 0.8)
axes[1].axis("tight")

axes[2].set_title("ax.axis('equal')", y = 0.7, x = 0.8)
axes[2].axis("equal")

plt.show()

# We can set one limit on an axis. The good practice is to set the limit after the plot
# is made (not before)

# Good -- setting the limits after the plot is made
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = plt.figaspect(0.5))
ax1.plot([-10, -5, 0, 5, 10, 15], [-1.2, 2, 3.5, -0.3, -4, 1])
ax2.scatter([-10, -5, 0, 5, 10, 15], [-1.2, 2, 3.5, -0.3, -4, 1])
ax1.set_ylim(bottom = -10)
ax2.set_xlim(right = 25)
plt.show()

# Bad -- setting the limits before the plot is made
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = plt.figaspect(0.5))
ax1.set_ylim(bottom = -10)
ax2.set_xlim(right = 25)
ax1.plot([-10, -5, 0, 5, 10, 15], [-1.2, 2, 3.5, -0.3, -4, 1])
ax2.scatter([-10, -5, 0, 5, 10, 15], [-1.2, 2, 3.5, -0.3, -4, 1])
plt.show()

## Legends
# In addition to labelling the figure and the axis, we can label the line / bar / point
# using legends (labels)
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [10, 20, 25, 30], label = "Philadelphia")
ax.plot([1, 2, 3, 4], [30, 23, 13, 4], label = "Boston")
ax.set(ylabel = "Temp (deg C)", xlabel = "Time", title = "A tale of 2 cities")
ax.legend(loc = "best")
plt.show()

# If we are plotting something that we don't want to appear in the legend, we can set that
# by not mentioning `label`
fig, ax = plt.subplots()
ax.bar([1, 2, 3, 4], [10, 20, 25, 30], label = "Foobar", align = "center", color = "lightblue")
ax.plot([1, 2, 3, 4], [10, 20, 25, 30], marker = "o", color = "darkred")
ax.legend(loc = "best")
plt.show()