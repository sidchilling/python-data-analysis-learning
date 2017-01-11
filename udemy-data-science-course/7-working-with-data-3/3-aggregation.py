import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

df = pd.read_csv('winequality-red.csv', sep = ';')
print df.head()

# average alcohol content of wines
print 'Avg. Alcohol Content: {}'.format(df['alcohol'].mean())

mean_alcohol_content = df['alcohol'].mean()
print df[df['alcohol'] > mean_alcohol_content].shape[0] # number of wines that have alcohol content greater than mean 
print df[df['alcohol'] < mean_alcohol_content].shape[0] # number of wines that have alchol content less than mean

# using custom aggregation
def max_to_min(arr):
    return arr.max() - arr.min()

wino = df.groupby(['quality'])
print wino.agg(max_to_min)
print wino.agg('mean') # same as using the mea() as in line #9

df['qual/alc ratio'] = df['quality'] / df['alcohol'] # adding a new column
print df.head()
print '\n\n'

# we can use pivot table instead of group
print df.pivot_table(index = 'quality')

# some visualization
df.plot(kind = 'scatter', x = 'quality', y = 'alcohol') # how quality is related to alcohol content
df.plot(kind = 'box', x = 'quality', y = 'alcohol')
plt.show()