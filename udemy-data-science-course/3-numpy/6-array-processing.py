from __future__ import division

import numpy as np
import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

points = np.arange(start = -5, stop = 5, step = 0.01)
print '{}'.format(points)

dx, dy = np.meshgrid(points, points)
print '{}'.format(dx)
print '{}'.format(dy)

print 'dx shape: {}'.format(dx.shape)
print 'dy shape: {}'.format(dy.shape)

z = np.sin(dx) + np.sin(dy)
print '{}'.format(z)
print 'z shape: {}'.format(z.shape)

plt.imshow(z)
plt.colorbar()
plt.title('Plot for sin(x) + sin(y)')
#plt.show() # uncomment this if you want the plot to show

# numpy where
A = np.array([1, 2, 3, 4])
B = np.array([100, 200, 300, 400])
condition = np.array([True, True, False, False])
answer = [(A_val if cond else B_val) for A_val, B_val, cond in zip(A, B, condition)]
print '{}'.format(answer)
answer = np.where(condition, A, B)
print '{}'.format(answer)

arr = np.random.randn(5, 5)
print '{}'.format(arr)
print '{}'.format(np.where(arr >0, arr, 0))

print '--- Sum ---'
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print '{}'.format(arr)
print 'Sum of all elements: {}'.format(arr.sum())
print 'Sum along columns: {}'.format(arr.sum(axis = 0))
print 'Sum along rows: {}'.format(arr.sum(axis = 1))
print 'Mean: {}'.format(arr.mean())
print 'Standard Deviation: {}'.format(arr.std())
print 'Variance: {}'.format(arr.var())

bool_arr = np.array([True, False, True])
for i in range(0, 5):
    print 'Any: {}'.format(bool_arr.any())
print 'All: {}'.format(bool_arr.all())

print '--- Sort ---'
arr = np.random.randn(5)
print 'Actual: {}'.format(arr)
arr.sort()
print 'Sorted: {}'.format(arr)

arr = np.random.randn(5, 5)
print '{}'.format(arr)
arr.sort()
print '{}'.format(arr)

countries = np.array(['France', 'Germany', 'USA', 'Russia', 'USA', 'Mexico', 'Germany'])
print '{}'.format(countries)
print '{}'.format(np.unique(countries))
print '{}'.format(np.in1d(['France', 'Germany', 'Sweden'], countries))