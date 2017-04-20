import matplotlib as mpl
mpl.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0.0, 5.0, 2)
print "T: {}".format(t)

plt.plot(t, t, "cyan", t, t**2, "r", t, t**3, "#45A8AD")
plt.show()