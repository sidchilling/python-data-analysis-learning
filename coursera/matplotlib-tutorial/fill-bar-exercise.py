import matplotlib as mpl
mpl.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt

# In this exercise, we will make the following figure -
# https://drive.google.com/file/d/0B0yTQeFvwuNjMVdNN1ZNcjJGZlE/view

## Data Generation (Provided)

yRaw = np.random.randn(1000).cumsum() + 15
xRaw = np.linspace(0, 24, yRaw.size)

# Get averages of every 100 samples
xPos = xRaw.reshape(-1, 100).min(axis = 1)
yAvg = yRaw.reshape(-1, 100).mean(axis = 1)
yErr = yRaw.reshape(-1, 100).ptp(axis = 1)

barWidth = xPos[1] - xPos[0]

# Make a made up future prediction with a fake confidence
xPred = np.linspace(0, 30)
yMaxPred = yAvg[0] + yErr[0] + 2.3 * xPred
yMinPred = yAvg[0] - yErr[0] + 1.2 * xPred

barColor, lineColor, fillColor = "wheat", "salmon", "lightblue"

## Now you're on your own

fig, ax = plt.subplots()

ax.plot(xRaw, yRaw, color = lineColor) # Make the Line Chart
ax.bar(xPos, yAvg, width = barWidth, color = barColor, yerr = yErr,
       ecolor = "gray", capsize = 3, edgecolor = "gray") # Make the Bar Chart with the errors
ax.fill_between(xPred, yMaxPred, yMinPred, color = fillColor) # Make the Fill

xlimMin = np.min([np.min(xRaw), np.min(xPos), np.min(xPred)]) + (barWidth / 2)
xlimMax = np.max([np.max(xRaw), np.max(xPos), np.max(xPred)])
ylimMin = np.min([np.min(yRaw), np.min(yAvg), np.min(yErr), np.min(yMaxPred), np.min(yMinPred)]) - 5
ylimMax = np.max([np.max(yRaw), np.max(yAvg), np.max(yErr), np.max(yMaxPred), np.max(yMinPred)])

ax.set(xlim = [xlimMin, xlimMax], ylim = [ylimMin, ylimMax], xlabel = "Minutes since class began",
       ylabel = "Snarkiness (snark units)", title = "Future Projection of Attitudes")

plt.show()