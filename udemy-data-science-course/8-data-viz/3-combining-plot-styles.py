import numpy as np
import pandas as pd

import matplotlib as mtp
mtp.use('TkAgg')

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

dataset = np.random.randn(100)

sns.distplot(dataset, bins = 25) # kernel density plot + histogram
plt.show()

sns.distplot(dataset, bins = 25, rug = True, hist = False) # Rug plot + kernel density plot
plt.show()

# Rug plot + Histogram + KDE Plot
sns.distplot(dataset, bins = 20, hist = True, kde = True, rug = True)
plt.show()

# Set params for individual plots
sns.distplot(dataset, bins = 25,
	     kde_kws = {'color' : 'indianred', 'label' : 'KDE Plot'},
	     hist_kws = {'color' : 'blue', 'label' : 'Histogram Plot'})
plt.show()

ser1 = pd.Series(dataset, name = 'my_data')
print ser1
sns.distplot(ser1, bins = 30)
plt.show()