from __future__ import division

import numpy as np

arr = np.array([1, 2, 3, 4, 5])
print '{}'.format(arr)

np.save('myarray', arr)
arr = np.arange(10)
print '{}'.format(arr)

arr = np.load('myarray.npy')
print '{}'.format(arr)

# save and load multiple arrays
arr1 = np.arange(10)
arr2 = np.arange(15)
print 'arr1: {}'.format(arr1)
print 'arr2: {}'.format(arr2)

np.savez('ziparray.npz', X = arr1, Y = arr2)

archived_arrays = np.load('ziparray.npz')
print 'X: {}'.format(archived_arrays['X'])
print 'Y: {}'.format(archived_arrays['Y'])

# Save as text files
arr = np.arange(20)
print '{}'.format(arr)
np.savetxt('mytextarray.txt', arr, delimiter = ',')
arr = np.loadtxt('mytextarray.txt', delimiter = ',')
print '{}'.format(arr)