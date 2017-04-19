import numpy as np

import matplotlib as mpl
mpl.use("TkAgg")

import matplotlib.pyplot as plt

def onPress(event):
    if not event.inaxes: return
    for line in event.inaxes.lines:
        if event.key == "t":
            visible = line.get_visible()
            line.set_visible(not visible)
    event.inaxes.figure.canvas.draw()

fig, ax = plt.subplots(1)
fig.canvas.mpl_connect("key_press_event", onPress)
ax.plot(np.random.rand(2, 20))
plt.show()