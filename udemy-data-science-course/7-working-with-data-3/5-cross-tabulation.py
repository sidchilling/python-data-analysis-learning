import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from StringIO import StringIO

# cross-tabulation is a special case of pivot table

data = """\
Sample Animal Intelligence
1 Dog Smart
2 Dog Smart
3 Cat Dumb
4 Cat Dumb
5 Dog Dumb
6 Cat Smart"""

df = pd.read_table(StringIO(data), sep = '\s+')
print df

print pd.crosstab(df.Animal, df.Intelligence, margins = True) # returns counts in pivot tables