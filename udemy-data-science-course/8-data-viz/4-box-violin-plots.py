import numpy as np
import pandas as pd

import matplotlib as mtp
mtp.use('TkAgg')

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

data1 = np.random.randn(100)
data2 = np.random.randn(100)

sns.boxplot(data = [data1, data2], whis = np.inf, orient = 'h')
plt.show()

# Normal distribution
data1 = stats.norm(0, 5).rvs(100) # 100 points between 0 and 5

# Two gamma distributions and concatenate together
data2 = np.concatenate([stats.gamma(5).rvs(50) - 1,
		       stats.gamma(5).rvs(50) * (-1)])

# Box plot both data1 and data2
sns.boxplot(data = [data1, data2], whis = np.inf)
plt.show()

# Violin plot combines a KDE plot and box plot
sns.violinplot(data = [data1, data2])
plt.show()

sns.violinplot(data = data1, inner = 'stick')
plt.show()