import numpy as np
import pandas as pd

from pandas import Series, DataFrame

df_left = DataFrame({'key' : ['X', 'Y', 'Z', 'X', 'Y'],
		     'data' : range(5)})
print df_left
df_right = DataFrame({'group_data' : [10, 20]},
		     index = ['X', 'Y'])
print df_right
print pd.merge(left = df_left, right = df_right, left_on = 'key',
	      right_index = True)

df_left_hr = DataFrame({'key1' : ['SF', 'SF', 'SF', 'LA', 'LA'],
			'key2' : [10, 20, 30, 20, 30],
			'data_set' : np.arange(5.)})
print df_left_hr
df_right_hr = DataFrame(np.arange(10).reshape(5, 2),
		       index = [['LA', 'LA', 'SF', 'SF', 'SF'], [20, 10, 10, 10, 20]],
		       columns = ['col_1', 'col_2'])
print df_right_hr
print pd.merge(left = df_left_hr, right = df_right_hr, left_on = ['key1', 'key2'],
	      right_index = True)

print '\n\n'

print df_left
print df_right
print df_left.join(df_right)