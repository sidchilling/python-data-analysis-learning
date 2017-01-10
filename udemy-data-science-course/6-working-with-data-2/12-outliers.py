import numpy as np
import pandas as pd
from pandas import DataFrame, Series

# How to find outliers in a dataset?

np.random.seed(12345) # seeding the random-number generator

df = DataFrame(np.random.randn(1000, 4))
print df.head()
print df.tail()
print df.describe()

col = df[0]
print col.head()

print col[np.abs(col) > 3] # show me cols with absolute value greater than 3
print df[(np.abs(df) > 3).any(1)]

# sets all values greater than 3 to 3
df[np.abs(df) > 3] = np.sign(df) * 3
print df.describe()

# the following should print an empty dataframe
print df[(np.abs(df) > 3).any(1)]