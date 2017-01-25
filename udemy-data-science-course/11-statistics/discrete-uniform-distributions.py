from __future__ import division

import numpy as np
from numpy.random import randn
import pandas as pd
from scipy import stats

import matplotlib as mtp
mtp.use('TkAgg')

import matplotlib.pyplot as plt
import seaborn as sns

# Checking Roll of a die
# Let's check out the Probability Mass Function
roll_options = [1, 2, 3, 4, 5, 6] # the possibile numbers with a die
tprob = 1 # total probability space
prob_roll = tprob / len(roll_options) # each roll has same probability (at least in a fair die)

uni_plot = sns.rugplot(a = roll_options, height = prob_roll,
		      c = 'indianred')
uni_plot.set_title('Probability Mass Function for Dice Roll')
plt.show()

# In the above example, `roll_options` is a discrete uniform distribution
# Let's calculate the mean and variance

np_mean = np.mean(roll_options)
formula_mean = (np.max(roll_options) + np.min(roll_options)) / 2
print 'NP Mean: {}, Formula Mean: {}'.format(np_mean, formula_mean)

np_variance = np.var(roll_options)
formula_variance = ((np.max(roll_options) - np.min(roll_options) + 1) ** 2) / 12
print 'NP Variance: {}, Formula Variance: {}'.format(np_variance, formula_variance)

## Use scipy create discrete uniform distribution
low, high = 1, 7
# Get mean and variance
mean, var = stats.randint.stats(low, high)
print 'The mean is: {}'.format(mean)

plt.bar(roll_options, stats.randint.pmf(roll_options, low, high))
plt.show()