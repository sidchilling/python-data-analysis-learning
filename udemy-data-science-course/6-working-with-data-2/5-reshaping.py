import numpy as np
import pandas as pd

from pandas import DataFrame, Series

df1 = DataFrame(np.arange(8).reshape(2, 4),
	       index = pd.Index(['LA', 'SF'], name = 'city'),
	       columns = pd.Index(['A', 'B', 'C', 'D'], name = 'letter'))
print df1

df_stack = df1.stack()
print df_stack
print df_stack.unstack()
print df_stack.unstack('letter')
print df_stack.unstack('city')

ser1 = Series([0, 1, 2], index = ['Q', 'X', 'Y'])
ser2 = Series([4, 5, 6], index = ['X', 'Y', 'Z'])
df = pd.concat([ser1, ser2], axis = 0, keys = ['Alpha', 'Beta']) # row-wise concat
df = df.unstack() # to convert to a DataFrame
print df

print df.stack() # removes the null-values
print df.stack(dropna = False) # keeps the null-values

