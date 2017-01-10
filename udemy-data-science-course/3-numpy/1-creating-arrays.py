# Section-3 Learning Numpy
# Lesson-7 Creating Arrays

import numpy as np

# create an array from a list
my_list1 = [1, 2, 3, 4]
my_array1 = np.array(my_list1)
print 'Value: {}'.format(my_array1)

my_list2 = [11, 22, 33, 44]
my_lists = [my_list1, my_list2]
my_array2 = np.array(my_lists)
print 'Value: {}'.format(my_array2)

# Find out the shape of the array (matrix). This returns a tuple (row, column)
print 'Shape: {}'.format(my_array2.shape)

# Find the type of the array, i.e. the type of the elements in the array
print 'Type: {}'.format(my_array2.dtype)

# Array of all zeros
my_zeros_array = np.zeros(5)
print 'Value {}'.format(my_zeros_array)
print 'Type: {}'.format(my_zeros_array.dtype)

# Array of all ones
my_ones_array = np.ones([5, 5]) # Matrix of 5x5
print 'Value: {}'.format(my_ones_array)
print 'Shape: {}'.format(my_ones_array.shape)
print 'Type: {}'.format(my_ones_array.dtype)

empty_array = np.empty(5)
print 'Value: {}'.format(empty_array)
print 'Shape: {}'.format(empty_array.shape)

# Identity Matrix (Always a Square matrix)
identity_matrix = np.eye(5) # 5x5 identity matrix
print 'Value: {}'.format(identity_matrix)
print 'Shape: {}'.format(identity_matrix.shape)

# Get evenly spaced numbers in an array
even_array = np.arange(5)
print 'Value: {}'.format(even_array)

even_array = np.arange(start = 5, stop = 50, step = 3)
print '{}'.format(even_array)