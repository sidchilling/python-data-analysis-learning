from __future__ import division

import matplotlib as mtp
mtp.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t
from scipy.stats import norm

x = np.linspace(start = -5, stop = 5, num = 1000)

mean = 0
std = 1

for num in range(3, 50, 10):

    rv = t(num)

    plt.plot(x, rv.pdf(x), c = 'green', label = 'T') # t-distribution
    plt.plot(x, norm.pdf(x, mean, std), c = 'red', label = 'Normal') # normal distribution
    plt.legend(loc = 'upper left')
    plt.title('For num = {}'.format(num))
    plt.show()