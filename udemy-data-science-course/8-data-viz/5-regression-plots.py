import numpy as np
import pandas as pd

import matplotlib as mtp
mtp.use('TkAgg')

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

tips = sns.load_dataset('tips')
print tips.head()

# shows a scatter plot with a liner regression fit line
sns.lmplot(x = 'total_bill', y = 'tip', data = tips,
	   scatter_kws = {'marker' : 'o', 'color' : 'indianred'},
	   line_kws = {'linewidth' : 1, 'color' : 'blue'})
plt.show()

# quadratic fit to the regression
sns.lmplot(x = 'total_bill', y = 'tip', data = tips, order = 4,
	   scatter_kws = {'marker' : 'o', 'color' : 'indianred'},
	   line_kws = {'linewidth' : 1, 'color' : 'blue'})
plt.show()

# only a scatter plot
sns.lmplot(x = 'total_bill', y = 'tip', data = tips, fit_reg = False)
plt.show()

tips['tip_pct'] = (tips['tip'] * 100) / tips['total_bill']
print tips.head()
sns.lmplot(x = 'size', y = 'tip_pct', data = tips)
plt.show()

# Adding Jitter
sns.lmplot(x = 'size', y = 'tip_pct', data = tips, x_jitter = 0.1)
plt.show()

# Adding tendency to tip by party size
sns.lmplot(x = 'size', y = 'tip_pct', data = tips, x_estimator = np.mean)
plt.show()

# Show how male and female differs in tips vs total bill
sns.lmplot(x = 'total_bill', y = 'tip_pct', data = tips,
	  hue = 'sex', markers = ['x', 'o'])
plt.show()

# Does day affect the tip percent
sns.lmplot(x = 'total_bill', y = 'tip_pct', data = tips,
	  hue = 'day')
plt.show()

# Local Regression
sns.lmplot(x = 'total_bill', y = 'tip_pct', data = tips,
	   lowess = True, line_kws = {'color' : 'black'})
plt.show()

# lmplt actually uses regplot (Regression Plot - More Generic)
sns.regplot(x = 'total_bill', y = 'tip_pct', data = tips)
plt.show()

# create a figure of two subplots
fig, (axis1, axis2) = plt.subplots(1, 2, sharey = True)
sns.regplot(x = 'total_bill', y = 'tip_pct', data = tips, ax = axis1)
sns.violinplot(x = 'size', y = 'tip_pct', data = tips,  ax = axis2)
plt.show()