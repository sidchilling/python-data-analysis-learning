import numpy as np
import pandas as pd
from pandas import Series, DataFrame

years = [1990, 1991, 1992, 2008, 2012, 1987, 1969, 2013, 2008, 1999]
decade_bins = [1960, 1970, 1980, 1990, 2000, 2010, 2020]

# category objects used for binning
decade_category = pd.cut(years, decade_bins)
print decade_category
print decade_category.categories
print pd.value_counts(decade_category)

# Binning based on min and max binning
print '\n\n'
two_bins = pd.cut(years, 2, precision = 1) # separate the `years` into 2 bins
print two_bins
print pd.value_counts(two_bins)
print '\n\n'
five_bins = pd.cut(years, 5, precision = 1)
print five_bins
print pd.value_counts(five_bins)