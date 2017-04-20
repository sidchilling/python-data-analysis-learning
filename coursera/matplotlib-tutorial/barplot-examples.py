import matplotlib as mlp
mlp.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt

# In this example, we will try to make the following graphs - 
# https://drive.google.com/file/d/0B0yTQeFvwuNjaFNYUWFkWTJ3VmM/view

fig, ax = plt.subplots(ncols = 3, figsize = plt.figaspect(0.5))

## The first vertical bar chart with error ticks
y = [1, 3, 4, 5.5, 3, 2]
errors = [0.2, 1, 2.5, 1, 1, 0.5]
x = np.arange(len(y))

print "X: {}".format(x)
print "Y: {}".format(y)
print "Errors: {}".format(errors)

ax[0].bar(x, y, color = "lightblue", edgecolor = "lightblue", yerr = errors,
          ecolor = "black", capsize = 4)
ax[0].set(xticks = [], yticks = [])

## The second horizontal bar
y = np.arange(8)
x1 = y + np.random.random(8) + 1
x2 = y + (3 * np.random.random(8)) + 1
print "Y: {}".format(y)
print "X1: {}".format(x1)
print "X2: {}".format(x2)

# Plot the positive horizontal bar
ax[1].barh(y, x1, color = "lightblue", edgecolor = "lightblue")
ax[1].barh(y, -x2, color = "salmon", edgecolor = "salmon")
ax[1].set(xticks = [], yticks = [])

## Making the graph for random rectangles
num = 10
left = np.random.randint(0, 10, num)
bottom = np.random.randint(0, 10, num)
width = np.random.random(num) + 0.5
height = np.random.random(num) + 0.5

print "Left: {}".format(left)
print "Bottom: {}".format(bottom)
print "Width: {}".format(width)
print "Height: {}".format(height)

ax[2].bar(left, height, width, bottom, color = "salmon", edgecolor = "salmon")
ax[2].margins(0.15)
ax[2].set(xticks = [], yticks = [])

plt.show()