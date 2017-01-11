import numpy as np
import pandas as pd
from pandas import Series, DataFrame


animals = DataFrame(np.arange(16).reshape(4, 4),
		   columns = ['W', 'X', 'Y', 'Z'],
		   index = ['Dog', 'Cat', 'Bird', 'Mouse'])
print animals
animals.ix[1 : 2, ['W', 'Y']] = np.nan
print animals

# group-by on a map
behavior_map = {
    'W' : 'good',
    'X' : 'bad',
    'Y' : 'good',
    'Z' : 'bad'
}
animal_col = animals.groupby(behavior_map, axis = 1) # row-wise
print animal_col.sum()

# group by on Series
behave_series = Series(behavior_map)
print animals.groupby(behave_series, axis = 1).count()

print animals.groupby(len).sum() # group by the length of the index

keys = ['A', 'B', 'A', 'B']
print animals.groupby([len, keys]).max()

hier_col = pd.MultiIndex.from_arrays([['NY', 'NY', 'NY', 'SF', 'SF'], 
				     [1, 2, 3, 1, 2]], names = ['city', 'sub_value'])
df_hr = DataFrame(np.arange(25).reshape(5, 5), columns = hier_col)
df_hr = df_hr * 100
print df_hr