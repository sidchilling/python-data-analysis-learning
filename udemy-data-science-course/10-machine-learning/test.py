import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import matplotlib as mtp
mtp.use('TkAgg')

import matplotlib.pyplot as plt
import seaborn as sns

df = DataFrame({
    'Factor' : ['Growth', 'Value'],
    'Weight' : [0.10, 0.20],
    'Variance' : [0.15, 0.35]
})
print df

tidy = (df.set_index('Factor').stack().reset_index().rename(columns = {'level_1' : 'Variable', 0 : 'Value'}))
print tidy