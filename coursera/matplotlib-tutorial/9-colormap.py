import matplotlib as mpl
mpl.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np

cmaps = [
        ("Sequential", ["binary", "Blues", "BuPu", "gist_yarg",
                        "GnBu", "Greens", "Greys", "Oranges", "OrRd",
                        "PuBu", "PuBuGn", "PuRd", "Purples", "RdPu",
                        "Reds", "YlGn", "YlGnBu", "YlOrBr", "YlOrRd"]),
        ("Sequential (2)", ["afmhot", "autumn", "bone", "cool", "copper",
                            "gist_gray", "gist_heat", "gray", "hot", "pink",
                            "spring", "summer", "winter"]),
        ("Diverging", ["BrBG", "bwr", "coolwarm", "PiYG", "PRGn", "PuOr",
                       "RdBu", "RdGy", "RdYlBu", "RdYlGn", "seismic"]),
        ("Qualitative", ["Accent", "Dark2", "hsv", "Paired", "Pastel1",
                         "Pastel2", "Set1", "Set2", "Set3", "spectral"]),
        ("Misc", ["gist_earth", "gist_ncar", "gist_rainbow",
                  "gist_stern", "jet", "brg", "CMRmap", "cubehelix",
                  "gnuplot", "gnuplot2", "ocean", "rainbow", "terrain",
                  "flag", "prism"])
        ]

nrows = max(len(cmapList) for cmapCategory, cmapList in cmaps)
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))

def plotColorGradients(cmapCategory, cmapList):
    fig, axes = plt.subplots(nrows = nrows)
    fig.subplots_adjust(top = 0.95, bottom = 0.01, left = 0.2, right = 0.99)
    axes[0].set_title(cmapCategory + " colormaps", fontsize = 14)

    for ax, name in zip(axes, cmapList):
        ax.imshow(gradient, aspect = "auto", cmap = plt.get_cmap(name))
        pos = list(ax.get_position().bounds)
        xText = pos[0] - 0.01
        yText = pos[1] + pos[3] / 2.0
        fig.text(xText, yText, name, va = "center", ha = "right", fontsize = 10)

    # Turn off all ticks & spines, not just the ones with colormaps
    for ax in axes:
        ax.set_axis_off()

for cmapCategory, cmapList in cmaps:
    plotColorGradients(cmapCategory, cmapList)

plt.show()