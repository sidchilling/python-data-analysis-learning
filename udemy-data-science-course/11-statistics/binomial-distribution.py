from __future__ import division

import scipy.misc as sc
from scipy.stats import binom
import numpy as np

import matplotlib as mtp
mtp.use('TkAgg')

import matplotlib.pyplot as plt

## Setup player A
p_A = 0.72 # probability of success for player A
n_A = 11 # number of shots
k = 6 # make 6 shots
comb_A = sc.comb(n_A, k)

answer_A = comb_A * (p_A ** k) * ((1 - p_A) ** (n_A - k))
print 'Probability of player A making {} success shots out of {} shots: {}%'.format(
	k, n_A, round(answer_A * 100, 2))

## For player B
p_B = 0.48
n_B = 15
comb_B = sc.comb(n_B, k)
answer_B = comb_B * (p_B ** k) * ((1 - p_B) ** (n_B - k))
print 'Probability of player B making {} success shots out of {} shots: {} %'.format(
	k, n_B, round(answer_B * 100, 2))

# Player B is a worse shooter but has higher probability to make 6 success shots
## What is k = 9? Will A's probability increase?

k = 9
comb_A = sc.comb(n_A, k)
comb_B = sc.comb(n_B, k)

answer_A = comb_A * (p_A ** k) * ((1 - p_A) ** (n_A - k))
answer_B = comb_B * (p_B ** k) * ((1 - p_B) ** (n_B - k))
print 'P(A|k = 9): {}%, P(B|k = 9): {}%'.format(round(answer_A * 100, 2), 
					     round(answer_B * 100, 2))

# Mean
mu_A = n_A * p_A
mu_B = n_B * p_B

print '{} : {}'.format(mu_A, mu_B)

# Standard Deviation
sigma_A = (n_A * p_A * (1 - p_A)) ** 0.5
sigma_B = (n_B * p_B * (1 - p_B)) ** 0.5

print 'Player A will make an average of {} +/- {} success shots per game'.format(
	int(round(mu_A, 0)), int(round(sigma_A, 0)))
print 'Player B will make an average of {} +/- {} success shots per game'.format(
	int(round(mu_B, 0)), int(round(sigma_B, 0)))

mean, var = binom.stats(n_A, p_A)
print 'Mean: {}, Var: {}'.format(mean, var)


# 10 coin flips 

n = 10
p = 0.5
x = range(n + 1)
# create the PMF
Y = binom.pmf(x, n, p)
print Y

# Plot Y
plt.plot(x, Y, 'o')
plt.title('Binomial Distribution PMF: 10 coin flips, Odds of success for Heads is p = 0.5', y = 1.08)
plt.xlabel('Number of Heads')
plt.ylabel('Probability')
plt.show()