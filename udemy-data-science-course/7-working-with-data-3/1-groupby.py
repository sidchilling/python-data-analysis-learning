import numpy as np
import pandas as pd
from pandas import Series, DataFrame

df = DataFrame({'k1' : ['X', 'X', 'Y', 'Y', 'Z'],
		'k2' : ['alpha', 'beta', 'alpha', 'beta', 'alpha'],
		'dataset1' : np.random.randn(5),
		'dataset2' : np.random.randn(5)})
print df

group1 = df['dataset1'].groupby(df['k1'])
print group1.mean()

cities = np.array(['NY', 'LA', 'LA', 'NY', 'NY'])
month = np.array(['Jan', 'Feb', 'Jan', 'Feb', 'Jan'])
print df['dataset1'].groupby([cities, month]).mean()
print df.groupby('k1').mean()
print df.groupby(['k1', 'k2']).mean()

print df
print df.groupby(['k1']).size()

# iterating over a group
for name, group in df.groupby(['k1']):
    print 'This is the {} group'.format(name)
    print group
    print '\n\n'

for (k1, k2), group in df.groupby(['k1', 'k2']):
    print 'key1 = {}, key2 = {}'.format(k1, k2)
    print group
    print '\n\n'

group_dict = dict(list(df.groupby(['k1'])))
print group_dict
print group_dict['X']

dataset2_group = df.groupby(['k1', 'k2'])[['dataset2']]
print df
print '\n\n'
print dataset2_group.mean()