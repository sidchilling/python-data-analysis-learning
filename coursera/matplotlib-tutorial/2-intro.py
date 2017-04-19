import matplotlib as mpl
mpl.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure() # Let's create a figure
plt.show() # Show the figure -- this generates an empty figure because we did not draw anything

# We can control the size of the figure using the `figsize` argument.
fig = plt.figure(figsize = (40, 10))
plt.show()

# Another useful utility function is `figaspect`
fig = plt.figure(figsize = plt.figaspect(2.0)) # Width = 2 * Height
plt.show()

# The most important thing is the `Axes` object. An `Axes` object must belong to a `figure` (and 
# only one `figure`). Majority of the drawing that we will do will be on an `Axes` object.
# An `Axes` object is made up of an x-axis and a y-axis.
fig = plt.figure()
# Adding a `subplot` to a figure is the most common way to get an axes object
ax = fig.add_subplot(111) # 111 will be explained later (basically 1 row and 1 column)
ax.set(xlim = [0.5, 4.5], ylim = [-2, 8], title = "An Example Axes", ylabel = "Y-Axis",
        xlabel = "X-Axis") # Use `set` to set properties to the axes
plt.show()

# Axes objects have setter function (as used above). We can set multiple properties at 
# once (as done above), or we can use the individual `set` methods for individual
# properties as below.
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([0.5, 4.5])
ax.set_ylim([-2, 8])
ax.set_title("An Example Axes")
ax.set_ylabel("Y-Axis")
ax.set_xlabel("X-Axis")
plt.show()

# Note: The `set` method applies to nearly all matplotlib objects, not just axes

# There are cases when we will need to use the individual `set` methods instead of passing kwargs
# to the global `set` method. Specially, when we need to set other options for a property, like below.
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set(xlim = [1, 10], ylim = [1, 5], ylabel = "Y-Axis", xlabel = "X-Axis")
ax.set_title("A Bigger Axes", size = 30)
plt.show()

# Most plotting happens on an axes, so we use the plotting methods available on an axes object.
# The 2 most important plotting methods are -- `plot` and `scatter`
# `plot` draws points connected by a line
# `scatter` draws unconnected points (optionally scaled or colored)

# Let's make a figure that contains contains 4 points connected by lines and 4 unconnected points
connectedPoints = [(1, 10), (2, 20), (3, 25), (4, 30)] # each tuple is a point in the cartesian system
unconnectedPoints = [(0.3, 11), (3.8, 25), (1.2, 9), (2.5, 26)]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([x[0] for x in connectedPoints], [x[1] for x in connectedPoints], color = "lightblue", linewidth = 3)
ax.scatter([x[0] for x in unconnectedPoints], [x[1] for x in unconnectedPoints], color = "darkgreen", marker = "^")
ax.set(xlim = [0.2, 5], ylim = [0, 35], xlabel = "X-Axis", ylabel = "Y-Axis",
        title = "Demonstrating Plot and Scatter")
plt.show()

# A figure can have multiple axes. The most common approach is to have multiple axes (smaller figures) on a
# regular grid system. We can create that using the `subplots` method, as below -
fig, axes = plt.subplots(nrows = 2, ncols = 2) # creates 4 axes (2 in each row)
# The axes returned is a numpy array. To get the axes object of one of the axes, we can use array indexing.
axes[0, 0].set(title = "Upper Left")
axes[0, 1].set(title = "Upper Right")
axes[1, 0].set(title = "Lower Left")
axes[1, 1].set(title = "Lower Right")
plt.show()

# We can also create a figure and one axes using the `subplots` method, but without passing any arguments
# So, the following code, which we saw earlier
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("One Axes")
plt.show()
# The above code can be replaced by the following much simpler code
fig, ax = plt.subplots()
ax.set_title("One Axes (Different Method)")
plt.show()

# We will use the second approach in most cases, as it's simpler. When we get to creating plots that do not 
# fit into a grid system, we will do the manual method of creating the figure and then adding axes manually.