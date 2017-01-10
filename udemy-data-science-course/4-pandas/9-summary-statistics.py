import numpy as np
import pandas as pd
import pandas.io.data as pdweb
import datetime

import matplotlib
matplotlib.use('TkAgg')

import seaborn as sns

import matplotlib.pyplot as plt
from pandas import Series, DataFrame


arr = np.array([[1, 2, np.nan], [np.nan, 3, 4]])
df = DataFrame(arr, index = ['A', 'B'], columns = ['One', 'Two', 'Three'])
print df

print df.sum() # sum on a dataframe (sum over columns)
print df.sum(axis = 1) # sum over rows

print df.min()
print df.min(axis = 1)
print df.idxmin()
print df.idxmax(axis = 1)

print df
print df.describe()

# Covariance and Correlation
prices = pdweb.get_data_yahoo(symbols = ['CVX', 'XOM', 'BP'],
			     start = datetime.datetime(2010, 1, 1),
			     end = datetime.datetime(2013, 1, 1))['Adj Close']
#print prices.head()

volume = pdweb.get_data_yahoo(symbols = ['CVX', 'XOM', 'BP'],
			     start = datetime.datetime(2010, 1, 1),
			     end = datetime.datetime(2013, 1, 1))['Volume']
#print volume.head()

returns = prices.pct_change()
correlation = returns.corr()
print correlation

prices.plot()
plt.show() # Un-comment if you want to show the graph

# Make a correlation plot
# This function is deprecated and should not be used
sns.linearmodels.corrplot(returns, annot = False, diag_names = False)
plt.show() # un-comment if you want to show the plot

# Get unique values
ser = Series(['a', 'a', 'b', 'b', 'b', 'c', 'd', 'd', 'd', 'd'])
print ser
print ser.unique()
print ser.value_counts()
