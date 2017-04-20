import matplotlib as mpl
mpl.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2 * np.pi * t)
plt.plot(t, s, lw = 2)

plt.annotate("Local Max", xy = (2, 1), xytext = (3, 1.5),
             arrowprops = {"facecolor": "black", "shrink": 0.05})

plt.ylim([-2, 2])
plt.show()

t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2 * np.pi * t)
plt.plot(t, s, lw = 2)

plt.annotate("Local Max", xy = (2, 1), xytext = (3, 1.5),
             arrowprops = {"facecolor": "red", "shrink": 0.05})
plt.ylim([-2, 2])
plt.show()