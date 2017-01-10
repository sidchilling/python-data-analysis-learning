import numpy as np
import pandas as pd

from pandas import DataFrame, Series
import pandas.util.testing as tm

# This is like doing Pivot Tables

tm.N = 3

def unpivot(frame):
    N, K = frame.shape

    data = {
	'value' : frame.values.ravel('F'),
	'variable' : np.asarray(frame.columns).repeat(N),
	'date' : np.tile(np.asarray(frame.index), K)
    }

    return DataFrame(data, columns = ['date', 'variable', 'value'])

df = unpivot(tm.makeTimeDataFrame())
print df

df_piv = df.pivot('date', 'variable', 'value')
print df_piv
