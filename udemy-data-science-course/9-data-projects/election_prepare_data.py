from __future__ import division

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import matplotlib as mtp
mtp.use('TkAgg')

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_style('whitegrid')

import requests
from StringIO import StringIO

from datetime import datetime

# Get the data from the URL
url = 'http://elections.huffingtonpost.com/pollster/2012-general-election-romney-vs-obama.csv'
source = requests.get(url = url).text
poll_data = StringIO(source)

poll_df = pd.read_csv(poll_data)

poll_df.info()
print poll_df.head()