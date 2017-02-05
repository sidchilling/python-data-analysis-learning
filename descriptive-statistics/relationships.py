import numpy as np
import pandas as pd
import matplotlib as mtp
mtp.use('TkAgg')

import matplotlib.pyplot as plt

data_l = [
    (7, 86), (8, 70), (6, 56), (5, 56), (6, 70), (7, 80), (6.5, 72),
    (8.5, 91), (6.5, 81), (7, 86)
]

data = []
for d in data_l:
    data.append({
	'hours_slept' : d[0],
	'memory_score' : d[1]
    })

df = pd.DataFrame(data)
print df

x = np.ravel(df['hours_slept'])
y = np.ravel(df['memory_score'])
m, b = np.polyfit(x = x, y = y, deg = 1)

df.plot(x = 'hours_slept', y = 'memory_score', kind = 'scatter')
plt.plot(x, m * x + b, '-')
plt.show()