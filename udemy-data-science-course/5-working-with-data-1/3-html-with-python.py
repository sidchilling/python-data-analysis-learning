import numpy as np
import pandas as pd

from pandas import Series, DataFrame

url = 'http://www.fdic.gov/bank/individual/failed/banklist.html'

dframe_list = pd.io.html.read_html(url)

df = dframe_list[0]
print df 
print df.columns
print df.columns.values