import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import matplotlib as mtp
mtp.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from sklearn.datasets import load_boston

boston = load_boston()
print boston.DESCR # scikit learn has given this decorator attribute

# Quick visualization of the data

plt.hist(x = boston.target, bins = 50)
plt.xlabel('Prices in $1000s')
plt.ylabel('Number of houses')
plt.show()

# Housing price vs number of rooms in the dwelling
plt.scatter(x = boston.data[:, 5], y = boston.target)
plt.ylabel('Price in $1000s')
plt.xlabel('No. of Rooms')
plt.show()

# Convert the data to the DataFrame
boston_df = DataFrame(boston.data)
boston_df.columns = boston.feature_names
boston_df['Price'] = boston.target
print boston_df.head()

sns.lmplot(x = 'RM', y = 'Price', data = boston_df)
plt.xlabel('No of Rooms')
plt.ylabel('Price in $1000s')
plt.show()