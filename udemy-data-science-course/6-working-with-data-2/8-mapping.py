import numpy as np
import pandas as pd
from pandas import DataFrame, Series

df = DataFrame({'city' : ['Alma', 'Brian Head', 'Fox Park'],
		'altitude' : [3158, 3000, 2762]})
print df

state_map = {
    'Alma' : 'Colorado',
    'Brian Head' : 'Utah',
    'Fox Park' : 'Wyoming'
}

df['state'] = df['city'].map(state_map)
print df