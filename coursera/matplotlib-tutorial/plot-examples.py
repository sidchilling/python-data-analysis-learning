import matplotlib as mlp
mlp.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt

# In this example, we are going to make the image in the following URL:
# https://drive.google.com/file/d/0B0yTQeFvwuNjeUp1eDh0VGpwbTQ/view

def lineGraph(x, ax):
    # Data for the first subplot (the different lines)
    lineSlopes = [0.2, 0.6, 1.0, 1.4, 1.8] # slopes of the lines
    ys = []
    intercept = 2
    for m in lineSlopes:
        ys.append((m * x) + intercept)

    colors = ["blue", "green", "red", "cyan", "purple"]

    for num in range(0, len(ys)):
        ax.plot(x, ys[num], color = colors[num], linewidth = 2)
        ax.set(xlim = [np.min(x), np.max(x)], xticks = [], yticks = [])

def curveGraph(x, ax):
    shifts = np.arange(4) # [0, 1, 2, 3]
    ys = []
    for shift in shifts:
        ys.append(np.cos(x) + shift)
    lineStyles = ["-", "--", ":", "-."]
    colors = ["blue", "green", "red", "cyan"]

    for num in range(0, len(ys)):
        ax.plot(x, ys[num], color = colors[num], linestyle = lineStyles[num])
        ax.set(xlim = [np.min(x), np.max(x)], xticks = [], yticks = [])

def markedGraph(x, ax):
    shifts = np.arange(3) # [0, 1, 2]
    ys = []
    for shift in shifts:
        ys.append(np.cos(x) + (shift * x))
    lineStyles = ["", "-", ":"]
    markerStyles = ["o", "^", "s"]
    colors = ["blue", "green", "red"]

    for num in range(0, len(ys)):
        ax.plot(x, ys[num], color = colors[num], linestyle = lineStyles[num],
                marker = markerStyles[num], markevery = 10)
        ax.set(xlim = [np.min(x), np.max(x)], xticks = [], yticks = [])

if __name__ == "__main__":
    x = np.linspace(0, 10, 100)
    fig, axes = plt.subplots(ncols = 3, figsize = plt.figaspect(0.5))
    lineGraph(x, axes[0])
    curveGraph(x, axes[1])
    markedGraph(x, axes[2])
    plt.show()
