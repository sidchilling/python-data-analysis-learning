import matplotlib as mpl
mpl.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt

# Let's start with plotting bar plots.
# We'll draw one figure with 2 subplots (one adjacent horizontally to the other)
# In the first, we will draw a vertical bar chart and in the other, a horizontal bar plot
# We will add some lines (splines) to the bar plots.

# Let's make some data first
np.random.seed(1)
x = np.arange(5)
y = np.random.randn(5)

print "X: {}".format(x)
print "Y: {}".format(y)

# Now, let's make the figure and the axes
fig, axes = plt.subplots(ncols = 2, figsize = plt.figaspect(1.0 / 2)) # We have 2 axes
vert_bars = axes[0].bar(x, y, color = "lightblue", align = "center")
axes[0].set(xlabel = "X-Axis", ylabel = "Y-Axis")
horiz_bars = axes[1].barh(x, y, color = "lightblue", align = "center")
axes[1].set(xlabel = "X-Axis", ylabel = "Y-Axis")

# Add Spine to the Axes
axes[0].axhline(0, color = "gray", linewidth = 2)
axes[1].axvline(0, color = "gray", linewidth = 2)

plt.show()

# Note: We held onto what the `bar` and `barh` methods return. These are called `Artists`. We
# can use them for special customizing that is not possible via the normal plotting methods.

# Let's try some customizations on the Artist
fig, ax = plt.subplots()
vert_bars = ax.bar(x, y, color = "lightblue", align = "center") # returns a list of artists
ax.axhline(0, color = "gray", linewidth = 1)
for bar in vert_bars:
    if bar.xy[1] < 0:
        bar.set(edgecolor = "darkred", color = "salmon", linewidth = 3)
plt.show()
