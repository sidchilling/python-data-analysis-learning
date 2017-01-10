import numpy as np
import pandas as pd

from pandas import Series, DataFrame

ser1 = Series([1, 2, 3, 4], index = ['A', 'B', 'C', 'D'])
print ser1

ser2 = ser1.reindex(['A', 'B', 'C', 'D', 'E', 'F'])
print ser2

print ser2.reindex(['A', 'B', 'C', 'D', 'E', 'F', 'G'], fill_value = 0)

ser3 = Series(['USA', 'Mexico', 'Canada'], index = [0, 5, 10])
print ser3
print ser3.reindex(range(15), method = 'ffill') # forward-fill
print ser3.reindex(range(15), method = 'bfill') # back-fill
print ser3.reindex(range(15), method = 'nearest') # nearest-fill_value

dframe = DataFrame(np.random.randn(25).reshape((5, 5)),
		  index = ['A', 'B', 'D', 'E', 'F'],
		  columns = ['col1', 'col2', 'col3', 'col4', 'col5'])
print dframe

# We forgot 'C' as index, so we will re-index
dframe1 = dframe.reindex(['A', 'B', 'C', 'D', 'E', 'F'])
print dframe1

dframe2 = dframe.reindex(columns = ['col1', 'col2', 'col3', 'col4', 'col5', 'col6'])
print dframe2

dframe3 = dframe.reindex(['A', 'B', 'C', 'D', 'E', 'F'],
			columns = ['col1', 'col2', 'col3', 'col4', 'col5', 'col6'])
print dframe3

print dframe.ix[['A', 'B', 'C', 'D', 'E', 'F'], ['col1', 'col2', 'col3', 'col4', 'col5', 'col6']]