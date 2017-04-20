import matplotlib as mpl
mpl.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np

bars = plt.bar([1, 2, 3, 4], [10, 12, 15, 17])
plt.setp(bars[0], hatch = "x", facecolor = "w")
plt.setp(bars[1], hatch = "xx-", facecolor = "orange")
plt.setp(bars[2], hatch = "+O.", facecolor = "c")
plt.setp(bars[3], hatch = "*", facecolor = "y")
plt.show()