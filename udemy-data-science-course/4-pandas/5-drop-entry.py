import numpy as np
import pandas as pd
from pandas import Series, DataFrame

ser1 = Series(np.arange(3), index = ['A', 'B', 'C'])
print ser1
print ser1.drop('B')

df = DataFrame(np.arange(9).reshape((3, 3)), index = ['SF', 'LA', 'NY'],
	      columns = ['Pop', 'Size', 'Year'])
print df
print df.drop('LA') # dropping a row
print df.drop('NY', axis = 0) # dropping a row
print df.drop('Size', axis = 1) # dropping a column
del df['Size'] # this permanently alters the dataframe
print df

