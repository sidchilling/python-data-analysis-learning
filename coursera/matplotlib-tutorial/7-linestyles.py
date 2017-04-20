import matplotlib as mpl
mpl.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0.0, 5.0, 0.2)
plt.plot(t, t, "-", t, t ** 2, "--", t, t ** 3, "-.", t, -t, ":")
plt.show()

fig, ax = plt.subplots(1, 1)
ax.bar([1, 2, 3, 4], [10, 20, 15, 13], ls = "dashed", ec = "r", lw = 5)
plt.show()