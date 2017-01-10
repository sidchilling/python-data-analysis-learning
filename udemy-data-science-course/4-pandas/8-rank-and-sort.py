import numpy as np
import pandas as pd

from pandas import DataFrame, Series

ser = Series(range(3), index = ['C', 'A', 'B'])
print ser
print ser.sort_index()
print ser.sort_values()

# Ranking
ser = Series(np.random.randn(10))
print ser
print ser.sort_values()
print ser.rank()
