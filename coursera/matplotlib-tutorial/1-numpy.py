import numpy as np

## Array Creation

# A numpy array has 3 properties - `shape`, `size`, and `ndim` (number of dimensions)
# 3 ways to create an array

def printProperties(arr):
    print "Shape: {}".format(arr.shape)
    print "Size: {}".format(arr.size)
    print "Number of Dimensions: {}".format(arr.ndim)
    print "\n"

if __name__ == "__main__":
    print "--- Method #1 ---"
    a = np.array([1, 2, 3])
    printProperties(a)

    print "--- Method #2 ---"
    x = np.arange(200)
    printProperties(x)

    print "--- Method #3 ---"
    y = np.random.rand(5, 80)
    printProperties(y)

