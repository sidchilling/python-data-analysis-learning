import numpy as np
import pandas as pd

from pandas import Series, DataFrame

df1 = DataFrame({'key' : ['X', 'Z', 'Y', 'Z', 'X', 'X'], 
		 'data_set_1' : np.arange(6)})
print df1

df2 = DataFrame({'key' : ['Q', 'Y', 'Z'], 'data_set_2' : [1, 2, 3]})
print df2

print pd.merge(df1, df2)
print pd.merge(df1, df2, on = 'key') # specifying which column to merge on
print pd.merge(df1, df2, on = 'key', how = 'left')
print pd.merge(df1, df2, on = 'key', how = 'right')
print pd.merge(df1, df2, on = 'key', how = 'outer')

df_left = DataFrame({'key1' : ['SF', 'SF', 'LA'],
		     'key2' : ['one', 'two', 'one'],
		     'left_data': [10, 20, 30]})
df_right = DataFrame({'key1' : ['SF', 'SF', 'LA', 'LA'],
		      'key2' : ['one', 'one', 'one', 'two'],
		      'right_data' : [40, 50, 60, 70]})
print df_left
print df_right

print pd.merge(df_left, df_right, on = ['key1', 'key2'], how = 'outer')