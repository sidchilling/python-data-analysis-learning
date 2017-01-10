import numpy as np
import pandas as pd

from pandas import DataFrame, Series

arr1 = np.arange(9).reshape(3, 3)
print arr1
print np.concatenate([arr1, arr1], axis = 1) # column wise
print np.concatenate([arr1, arr1], axis = 0) # row wise

# concatenate in pandas
ser1 = Series([0, 1, 2], index = ['T', 'U', 'V'])
ser2 = Series([3, 4], index = ['X', 'Y'])
print ser1
print ser2
print pd.concat([ser1, ser2], axis = 1)
print pd.concat([ser1, ser2], axis = 0)
print pd.concat([ser1, ser2], axis = 0, keys = ['cat1', 'cat2'])

df1 = DataFrame(np.random.rand(4, 3), columns = ['X', 'Y', 'Z'])
df2 = DataFrame(np.random.randn(9).reshape(3, 3), columns = ['Y', 'Q', 'X'])
print df1
print df2
print pd.concat([df1, df2], axis = 0) # row-wise concat
print pd.concat([df1, df2], axis = 1) # column-wise concat
print pd.concat([df1, df2], axis = 0, ignore_index = True) # row-wise concact, but ignore index