from __future__ import division

import numpy as np

arr = np.arange(11)
print '{}'.format(arr)

print '{}'.format(np.sqrt(arr))
print '{}'.format(np.exp(arr))

A = np.random.randn(10)
print '{}'.format(A)
B = np.random.randn(10)
print '{}'.format(B)

# Binary Functions
print '--- Adding 2 Arrays ---'
print '{}'.format(np.add(A, B))

print '--- Find Max of each index between two arrays ---'
print '{}'.format(np.maximum(A, B))

print '--- Find Min of each index between two arrays ---'
print '{}'.format(np.minimum(A, B))