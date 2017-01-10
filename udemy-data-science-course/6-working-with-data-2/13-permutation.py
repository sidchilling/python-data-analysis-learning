import numpy as np
import pandas as pd
from pandas import Series, DataFrame

df = DataFrame(np.arange(16).reshape(4, 4))
print df

blender = np.random.permutation(4) # a random permutation
print blender

# take random permutation of columns from the dataframe
print df.take(blender) # row-wise permutation
print df.take(blender, axis = 1) # column-wise permutation

# pick numbered marbles from the box based on the permutation
box = np.array([1, 3, 2, 4, 6, 5])
shaker = np.random.randint(0, len(box), size = 20)
print shaker
print box.take(shaker)
