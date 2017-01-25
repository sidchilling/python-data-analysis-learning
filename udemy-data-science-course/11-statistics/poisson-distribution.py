from __future__ import division

from math import exp
from math import factorial
from scipy import stats
import numpy as np

import matplotlib as mtp
mtp.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

'''
Say, McDonald's has a lunch rush from 12:30pm to 1pm. From previous days' data,
we know that on an average 10 customers enter during 12:30pm and 1pm. What is the
probability that exactly 7 customers enter during lunch rush? What is the probability
that more than 10 customers enter?
'''

lamb = 10 # In Poisson distribution, lambda is the mean -- in this case 10.
k = 7 # number of customers that must enter (the first question)

prob = (lamb ** k) * exp(-lamb) / factorial(k)
print 'The probability that {} customers enter during lunch rush: {}%'.format(k,
								round(prob * 100, 2))

## Make the above PMF using Poisson
mu = 10
mean, var = stats.poisson.stats(mu)
odds_seven = stats.poisson.pmf(7, mu)

print 'Mean: {}, Var: {}'.format(mean, var)
print 'The probability that {} customers enter during lunch rush: {}'.format(k,
								    round(odds_seven* 100, 2))

## We need to see the entire distribution to answer the second question
k = np.arange(30) # PMF for customers all the way to 30
pmf_poisson = stats.poisson.pmf(k, lamb)

plt.bar(k, pmf_poisson)
plt.show()

# To find the probability of more than 10 customers, we need to sum above 10.
# Use a cumulative distribution function (CDF)

k, mu = 10, 10
prob_up_to_ten = stats.poisson.cdf(k, mu)
print 'Probability of 10 or less customers: {}%'.format(round(prob_up_to_ten * 100, 2))
prob_ten_or_more = 1 - prob_up_to_ten
print 'Probability of more than 10 customers: {}%'.format(round(prob_ten_or_more * 100, 2))