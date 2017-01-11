import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from scipy import stats

import matplotlib as mpl
mpl.use('TkAgg')

import matplotlib.pyplot as plt
import seaborn as sns

dataset1 = np.random.randn(100)
# plot the histogram
plt.hist(dataset1)
plt.show()

dataset2 = np.random.randn(80)
# plot the histogram
plt.hist(dataset2, color = 'indianred')
plt.show()

# plot both the datasets in a histogram on the same plot
plt.hist(dataset1, normed = True, color = 'indianred', alpha = 0.5,
	bins = 20)
plt.hist(dataset2, normed = True, alpha = 0.5, bins = 20)
plt.show()

# using seaborn
data1 = np.random.randn(1000)
data2 = np.random.randn(1000)
sns.jointplot(data1, data2)
plt.show()

sns.jointplot(data1, data2, kind = 'hex')
plt.show()