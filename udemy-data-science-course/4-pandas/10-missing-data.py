import numpy as np
import pandas as pd
from pandas import Series, DataFrame

data = Series(['one', 'two', np.nan, 'four'])
print data
print data.isnull() # find null values
print data.dropna() # drop null values

df = DataFrame([[1, 2, 3], [np.nan, 5, 6], [7, np.nan, 9], [np.nan, np.nan, np.nan]])
print df
print df.isnull()
print df.dropna() # drops all the rows with at least one NaN

print df.dropna(how = 'any') # same as above
print df.dropna(how = 'all')

print df.dropna(axis = 1, how = 'any')
print df.dropna(axis = 1, how = 'all')

df = DataFrame([[1, 2, 3, np.nan], [2, np.nan, 5, 6], [np.nan, 7, np.nan, 9],
	       [1, np.nan, np.nan, np.nan]])
print df
print df.dropna(thresh = 2) # Drop all rows that does not contain at least 2 non-nan datapoints
print df.dropna(thresh = 3, axis = 1) # Drop all columns that do not contain at least 3 non-nan datapoints

print df.fillna(1)
print df.fillna({0 : 0, 1 : 1, 2 : 2, 3 : 3})

# in-place changing the dataframe
print df
df.fillna({0 : 0, 1 : 1, 2 : 2, 3 : 3}, inplace = True)
print df