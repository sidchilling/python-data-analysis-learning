from __future__ import division

import numpy as np
from scipy import stats

import matplotlib as mtp
mtp.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

mean = 0
std = 1

X = np.arange(-4, 4, 0.01)
Y = stats.norm.pdf(X, mean, std)

plt.plot(X, Y)
plt.show()

# use numpy to make the normal distribution
mu, sigma = 0, 0.1

# now grab 1000 random numbers from the normal distribution
for num in [1000, 5000, 10000, 50000, 100000, 500000]:
    norm_set = np.random.normal(mu, sigma, num)

    plt.hist(norm_set, bins = 50)
    plt.title('Num points: {}'.format(num))
    plt.show()