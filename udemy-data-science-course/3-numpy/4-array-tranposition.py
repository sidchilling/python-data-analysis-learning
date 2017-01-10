from __future__ import division

import numpy as np

arr = np.arange(50).reshape((10, 5)) # reshape to a 10x5 matrix
print '{}'.format(arr)
print 'Shape: {}'.format(arr.shape)

print '--- Transpose---'
print '{}'.format(arr.T)
print 'Shape: {}'.format(arr.T.shape)

print '--- Dot Product ---'
dot_product_arr = np.dot(arr.T, arr)
print '{}'.format(dot_product_arr)
print 'Shape: {}'.format(dot_product_arr.shape)

print '--- 3D Matrix ---'
arr_3d = np.arange(50).reshape((5, 5, 2))
print '{}'.format(arr_3d)
print '{}'.format(arr_3d.shape)
arr_3d_transposed = arr_3d.T
print '{}'.format(arr_3d_transposed)

print '--- Swap Axes ---'
arr = np.array([[1, 2, 3]])
print '{}'.format(arr)
print '{}'.format(arr.shape)
print '{}'.format(arr.T)
print '{}'.format(arr.swapaxes(0, 1))