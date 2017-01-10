import numpy as np
import pandas as pd

from pandas import Series, DataFrame

ser = Series(np.random.randn(6), index = [[1, 1, 1, 2, 2, 2],
					 ['A', 'B', 'C', 'A', 'B', 'C']])
print ser
print ser.ix[1]
print ser.ix[2]

print ser.ix[:, 'A']
print ser.ix[1, 'B']

df = ser.unstack()
print df

df = DataFrame(np.arange(16).reshape(4, 4),
	      index = [['A', 'A', 'B', 'B'], [1, 2, 1, 2]],
	      columns = [['NY', 'NY', 'LA', 'SF'], ['Cold', 'Hot', 'Cold', 'Cold']])
df.index.names = ['Index #1', 'Index #2']
df.columns.names = ['Cities', 'Temp']
print df

# Swap the columns
df = df.swaplevel('Cities', 'Temp', axis = 1)
print df

print df.sum(level = 'Temp', axis = 1)
