import numpy as np
import pandas as pd

from pandas import Series, DataFrame

df = pd.read_csv('data.csv', header = None)
print df

# read_table is a more generic method
df = pd.read_table('data.csv', sep = ',', header = None)
print df

# read specific number of rows from the file
print pd.read_csv('data.csv', header = None, nrows = 2)

df.to_csv('saved_file.csv')