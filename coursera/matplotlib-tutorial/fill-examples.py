import matplotlib as mlp
mlp.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt

# In this example, we will make the following graphs-
# https://drive.google.com/file/d/0B0yTQeFvwuNjSWNfMHFDMmdBV0k/view

def fillData():
    t = np.linspace(0, 2 * np.pi, 100)
    r = np.random.normal(0, 1, 100).cumsum()
    r -= r.min()
    return r * np.cos(t), r * np.sin(t)

def fillBetween():
    x = np.linspace(0, 50, 100) # 100 linearlly spaced points in [0, 50)
    intercept = 2
    slope = 1
    y = slope * x + intercept
    err1 = np.random.random_sample(100) + np.random.random_sample(100)
    y1 = y + err1
    err2 = np.random.random_sample(100) + np.random.random_sample(100)
    y2 = y - err2

    print "X: {}".format(x)
    print "Y1: {}".format(y1)
    print "Y2: {}".format(y2)

    print "X Shape: {}".format(x.shape)
    print "Y1 Shape: {}".format(y1.shape)
    print "Y2 Shape: {}".format(y2.shape)

    return x, y1, y2


fig, axes = plt.subplots(ncols = 3, figsize = plt.figaspect(0.5))

## Make the first graphs
x, y = fillData()
print "X: {}".format(x)
print "Y: {}".format(y)
axes[0].fill(x, y, color = "lightblue")
axes[0].set(xticks = [], yticks = [])

x, y1, y2 = fillBetween()
axes[1].fill_between(x, y1, y2, color = "#FF9A02")
axes[1].set(xticks = [], yticks = [], xlim = (-5, np.max(x) + 4))

x = np.linspace(0, 50, 100)
y = 2 * np.sin(x)
print "Y Shape: {}".format(y.shape)
print "X Shape: {}".format(x.shape)
axes[1].fill_betweenx(x, -y, color = "red", where = y > 0)
axes[1].fill_betweenx(x, -y, color = "blue", where = y <= 0)
axes[1].margins(0.15)

x1 = np.linspace(0, 50, 100)
y1 = 2 * np.sin(x)
y2 = 2 * np.cos(x)
axes[1].fill_between(x, y1, y2, color = "lightblue", where = y1 > y2)
axes[1].fill_between(x, y1, y2, color = "forestgreen", where = y1 <= y2)

x = np.linspace(0, 10, 100)
y = np.random.normal(0, 1, (5, 100))
y = y.cumsum(axis = 1)
y -= y.min(axis = 0, keepdims = True)
axes[2].stackplot(x, y.cumsum(axis = 0), alpha = 0.5)
axes[2].set(xticks = [], yticks = [])

plt.show()
