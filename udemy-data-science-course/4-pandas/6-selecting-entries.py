import numpy as np
import pandas as pd

from pandas import Series, DataFrame

ser1 = Series(np.arange(3), index = ['A', 'B', 'C'])
ser1 = 2 * ser1
print ser1
print ser1['B']
print ser1[1]
print ser1[0 : 3]
print ser1[['A', 'C']]
print ser1[ser1 > 3]

ser1[ser1 > 3] = 10
print ser1

df = DataFrame(np.arange(25).reshape((5, 5)), index = ['NYC', 'LA', 'SF', 'DC', 'Chi'],
	      columns = ['A', 'B', 'C', 'D', 'E'])
print df

print df['B']
print df[['B', 'E']]
print df[df['C'] > 8]
print df.ix['LA']
print df > 10