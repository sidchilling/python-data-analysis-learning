import numpy as np
import pandas as pd
from pandas import Series, DataFrame

my_ser = Series([1, 2, 3, 4], index = ['A', 'B', 'C', 'D'])
print '{}'.format(my_ser)

my_index = my_ser.index
print '{}'.format(my_index)
print '{}'.format(my_index[2])

try:
    my_index[0] = 'Z'
except:
    print 'Indexes are immutable'