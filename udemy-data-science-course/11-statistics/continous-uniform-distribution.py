from __future__ import division

from scipy import stats
import numpy as np

import matplotlib as mtp
mtp.use('TkAgg')

import matplotlib.pyplot as plt

a = 19 # lower bound time
b = 27 # upper bound time
fx = 1 / (b - a)
print 'The probability density function is: {}'.format(fx)

var = ((b - a) ** 2) / 12
print 'The variance of the continous function density is: {}'.format(var)

## What is the probability that the taxi ride will take at least 25 mins?
fx_1 = 27 / (b - a) # f(27)
fx_2 = 25 / (b - a) # f(25)
ans = fx_1 - fx_2
print 'The probability that the taxi ride will take at least 25 mins: {}%'.format(int(ans * 100))

## Using Scipy to do the above

# Let's set up A and B
A, B = 0, 5
x = np.linspace(start = A, stop = B, num = 100) # setup x as 100 linearly spaced points between A and B
rv = stats.uniform(loc = A, scale = B) # use uniform 
# plot the PDF of the uniform distribution
plt.plot(x, rv.pdf(x))
plt.show()