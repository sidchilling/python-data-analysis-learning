import numpy as np
import pandas as pd
from pandas import Series, DataFrame

obj = Series([3, 6, 9, 12])
print '{}'.format(obj)
print 'Values: {}'.format(obj.values)
print 'Index: {}'.format(obj.index)

ww2_cas = Series([8700000, 4300000, 3000000, 2100000, 400000], index = ['USSR',
				    'Germany', 'China', 'Japan', 'USA'])
print '{}'.format(ww2_cas)
print 'USA Value: {}'.format(ww2_cas['USA'])

# Check which countries had casualties greater than 4mil
print '{}'.format(ww2_cas[ww2_cas > 4000000])

# Whether an index is in a series
print '{}'.format('USSR' in ww2_cas)
print '{}'.format('India' in ww2_cas)

# Convert a Series to a dict
ww2_dict = ww2_cas.to_dict()
print '{}'.format(ww2_dict)

# Convert a dict to a Series
ww2_series = Series(ww2_dict)
print '{}'.format(ww2_series)

countries = ['China', 'Germany', 'Japan', 'USA', 'USSR', 'Argentina']
obj2 = Series(ww2_dict, index = countries)
print '{}'.format(obj2)
print '{}'.format(pd.isnull(obj2))
print '{}'.format(pd.notnull(obj2))

print '--- Adding 2 Series ---'
print '{}'.format(ww2_series)
print '{}'.format(obj2)
print '{}'.format(ww2_series + obj2)

print '--- Naming a Series ---'
obj2.name = 'World War-II Casualties'
obj2.index.name = 'Countries'
print '{}'.format(obj2)
