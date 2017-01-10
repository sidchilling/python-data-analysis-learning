import numpy as np
import pandas as pd

from pandas import Series, DataFrame

ser1 = Series([0, 1, 2], index = ['A', 'B', 'C'])
ser2 = Series([3, 4, 5, 6], index = ['A', 'B', 'C', 'D'])
ser3 = ser1 + ser2

print ser1
print ser2
print ser3

df1 = DataFrame(np.arange(4).reshape((2, 2)), index = ['NYC', 'LA'], columns = list('AB'))
df2 = DataFrame(np.arange(9).reshape((3, 3)), index = ['NYC', 'SF', 'LA'],
	       columns = list('ACD'))
print df1
print df2
print df1 + df2
print df1.add(df2, fill_value = 0)

# Operations between a dataframe and a series
ser = df2.ix[0]
print ser

print df2
print df2 - ser
print df2 + ser