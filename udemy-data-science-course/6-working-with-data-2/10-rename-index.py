import numpy as np
import pandas as pd
from pandas import DataFrame, Series

df = DataFrame(np.arange(12).reshape(3, 4),
	      index = ['NY', 'LA', 'SF'],
	      columns = ['A', 'B', 'C', 'D'])
print df
df.index = df.index.map(str.lower)
print df

print df.rename(index = str.title, columns = str.lower) # renaming indexes using `map`
print df.rename(index = {'ny' : 'New York'},
		columns = {'A' : 'Alpha'}) # Renaming some indexes and columns

# Edit the dataframe inplace
df.rename(index = {'ny' : 'New York', 'la' : 'Los Angeles', 'sf' : 'San Francisco'},
	 inplace = True)
print df