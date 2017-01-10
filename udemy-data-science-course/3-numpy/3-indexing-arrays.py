from __future__ import division

import numpy as np

arr = np.arange(0, 11)
print '{}'.format(arr)
print '{}'.format(arr[8])
print '{}'.format(arr[1 : 5])
print '{}'.format(arr[0 : 5])

arr[0 : 5] = 100 # set all the elements from 0 to 5 to 100
print '{}'.format(arr)

arr = np.arange(0, 11)
print '{}'.format(arr)

slice_of_arr = arr[0 : 6]
print '{}'.format(slice_of_arr)
slice_of_arr[:] = 999
print '{}'.format(slice_of_arr)
print '{}'.format(arr) # slice_of_arr is only a view of arr, so arr will be changed

arr = np.arange(0, 11)
slice_of_arr_copy = arr[0 : 6].copy()
print '{}'.format(slice_of_arr_copy)
slice_of_arr_copy[:] = 999
print '{}'.format(slice_of_arr_copy)
print '{}'.format(arr)

arr_2d = np.array(([5, 10, 15], [20, 25, 30], [35, 40, 45]))
print '{}'.format(arr_2d)

# Show a row
print 'Row #1: {}'.format(arr_2d[1])
print 'Row #1, Col #0: {}'.format(arr_2d[1][0])

# Row and column slicing
print '{}'.format(arr_2d[:2, 1:])
print '{}'.format(arr_2d[1:,0:])

# Column Slicing
column_sliced = arr_2d[:, 1] # Gives the column slice but in a row array
print 'Column Slicing: {}'.format(column_sliced)
print 'Shape: {}'.format(column_sliced.shape)

arr2d = np.zeros((10, 9))
print '{}'.format(arr2d)
print 'Length: {}'.format(arr2d.shape[0])

for i in range(0, arr2d.shape[0]):
    arr2d[i] = i

print '{}'.format(arr2d)

# Fancy Indexing
# Get random rows from the 2D-array
print '{}'.format(arr2d[[2, 5, 9, 3]])