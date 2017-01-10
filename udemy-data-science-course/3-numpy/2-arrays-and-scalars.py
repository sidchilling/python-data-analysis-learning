from __future__ import division
import numpy as np

res = 5 / 2
print '{}'.format(res)

arr1 = np.array([[1, 2, 3, 4], [8, 9, 10, 11]])
print '{}'.format(arr1)

# Multiply each element of the array separately
print '{}'.format(arr1 * arr1)

# Subtaction
print '{}'.format(arr1 - arr1)

# Scalar Division
print '{}'.format(1 / arr1)

# Scalar Exponent
print '{}'.format(arr1 ** 2)