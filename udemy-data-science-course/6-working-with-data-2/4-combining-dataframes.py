import numpy as np
import pandas as pd

from pandas import DataFrame, Series

ser1 = Series([2, np.nan, 4, np.nan, 6, np.nan],
	     index = ['Q', 'R', 'S', 'T', 'U', 'V'])
print ser1
ser2 = Series(np.arange(len(ser1)), dtype = np.float64,
	     index = ['Q', 'R', 'S', 'T', 'U', 'V'])
print ser2

print Series(np.where(pd.isnull(ser1), ser2, ser1), index = ser1.index)
print ser1.combine_first(ser2) # short way to do the same thing as immediately above

df_odds = DataFrame({'X' : [1., np.nan, 3., np.nan],
		     'Y' : [np.nan, 5., np.nan, 7.],
		     'Z' : [np.nan, 9., np.nan, 11.]})
df_evens = DataFrame({'X' : [2., 4., np.nan, 6., 8.],
		      'Y' : [np.nan, 10., 12., 14., 16.]})
print df_odds
print df_evens
print df_odds.combine_first(df_evens)

# The following will not work because the shape of the dataframes are different
try:
    print DataFrame(np.where(pd.isnull(df_odds), df_evens, df_odds),
		    index = df_odds.index)
except Exception as e:
    print '{}'.format(e)


# We can use the np.where methodology to combine dataframes if they are
# of the same shape
df_evens = DataFrame({'X' : [2., 4., np.nan, 6.],
		      'Y' : [np.nan, 10., 12., 14.],
		      'Z' : [18., np.nan, 20., 22.]})
print df_evens

print DataFrame(np.where(pd.isnull(df_odds), df_evens, df_odds),
	       index = df_odds.index)