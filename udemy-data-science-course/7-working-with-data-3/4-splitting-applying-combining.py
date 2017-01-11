import numpy as np
import pandas as pd
from pandas import DataFrame, Series

import time

# split, apply, and combine

df = pd.read_csv('winequality-red.csv', sep = ';')
print df.head()
print '\n\n'

# highest alcohol content wine for each quality range

start_time = time.time()
for i in range(0, 100):
    print df.groupby(['quality'])['alcohol'].max()
end_time = time.time()
short_method_time = end_time - start_time

def ranker(df):
    df['alc_content_rank'] = np.arange(len(df)) + 1
    return df

start_time = time.time()
for i in range(0, 100):
    df.sort('alcohol', ascending = False, inplace = True)
    df = df.groupby('quality').apply(ranker)

    num_of_qual = df['quality'].value_counts()

    print df[df.alc_content_rank == 1].head(len(num_of_qual))[['quality', 'alcohol']]
end_time = time.time()
long_method_time = end_time - start_time

print 'Short Method Time: {}, Long Method Time: {}'.format(short_method_time, long_method_time)
if short_method_time < long_method_time:
    print 'Short Method Works faster'
else:
    print 'Long Method Works faster'