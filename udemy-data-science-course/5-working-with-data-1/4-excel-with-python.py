import numpy as np
import pandas as pd

from pandas import Series, DataFrame

xlfile = pd.ExcelFile('data-file.xlsx')
df = xlfile.parse('Sheet1')
print df