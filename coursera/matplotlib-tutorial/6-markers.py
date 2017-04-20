import matplotlib as mpl
mpl.use("TKAgg")

import matplotlib.pyplot as plt
import numpy as np

xs, ys = np.mgrid[:4, 9:0:-1]
markers = [".", "+", ",", "x", "o", "D", "d", "", "8", "s", "p", "*", "|", "_", "h", "H", 0, 4, "<", "3",
           1, 5, ">", "4", 2, 6, "^", "2", 3, 7, "v", "1", "None", None, " ", ""]
descripts = ["point", "plus", "pixel", "cross", "circle", "diamond", "thin diamond", "",
            "octagon", "square", "pentagon", "star", "vertical bar", "horizontal bar", "hexagon 1", "hexagon 2",
            "tick left", "caret left", "triangle left", "tri left", "tick right", "caret right", "triangle right", "tri right",
            "tick up", "caret up", "triangle up", "tri up", "tick down", "caret down", "triangle down", "tri down",
            "Nothing", "Nothing", "Nothing", "Nothing"]
fig, ax = plt.subplots(1, 1, figsize = (14, 4))
for x, y, m, d in zip(xs.T.flat, ys.T.flat, markers, descripts):
    ax.scatter(x, y, marker = m, s = 100)
    ax.text(x + 0.1, y - 0.1, d, size = 14)
ax.set_axis_off()
plt.show()

t = np.arange(0.0, 5.0, 0.2)
plt.plot(t, t, "+", t, t ** 2, "v", t, t ** 3, "H")
plt.show()