import numpy as np
import pandas as pd
from pandas import DataFrame, Series

ser1 = Series([1, 2, 3, 4, 1, 2, 3, 4])
print ser1

print ser1.replace(to_replace = 1, value = np.nan)
print ser1.replace(to_replace = [1, 4], value = [100, 400])
print ser1.replace({4 : np.nan, 2 : 200})