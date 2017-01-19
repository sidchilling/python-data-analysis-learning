from __future__ import division

import math

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib as mtp
mtp.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

import statsmodels.api as sm

## Plot the logistic function 
def logistic(t):
    return 1.0 / (1 + math.exp((-1.0) * t))

# Set t from -6 to 6 (500 elements, linearly spaced)
t = np.linspace(start = -6, stop = 6, num = 500)
# Set up y values using list comprehension
y = np.array([logistic(el) for el in t])
# Plot
plt.plot(t, y)
plt.title('Logistic Function')
plt.show()

# From the dataset, we want to answer the question -
# Given features, will a woman have an extra-martial affair?

# Load the data from statsmodels
df = sm.datasets.fair.load_pandas().data
print df.head()

def affair_check(x):
    if x != 0: return 1
    return 0

df['Had_Affair'] = df['affairs'].apply(affair_check)
print df.head()
print 'Data Length: {}'.format(len(df))

# Group by Had Affair
print df.groupby('Had_Affair').mean()

# Visualize by Age
sns.factorplot('age', data = df, hue = 'Had_Affair', palette = 'coolwarm',
	      kind = 'count')
plt.show() 
# Turns out that younger women have more tendency to have an affair and it 
# becomes even or comparable as they get older

# Visualize by Years Married
sns.factorplot('yrs_married', data = df, hue = 'Had_Affair', palette = 'coolwarm',
	      kind = 'count')
plt.show()

# Visualize by number of children
sns.factorplot('children', data = df, hue = 'Had_Affair', palette = 'coolwarm',
	      kind = 'count')
plt.show()

# Visualize by occupation
sns.factorplot('occupation', data = df, hue = 'Had_Affair', palette = 'coolwarm',
	      kind = 'count')
plt.show()

## Data Preparation for Logistic Regression

# Important - Occupation is a categorical data column (even though it is a number,
# but they are discrete values and not continous variable)
# So, we have to change the categorical data into multiple columns
# If we don't do it, regression will be confused that the feature will 
# take continous values along a spectrum

occ_dummies = pd.get_dummies(data = df['occupation'])
hus_occ_dummies = pd.get_dummies(data = df['occupation_husb'])
print occ_dummies.head()

# Renaming the columns in the above dummies to make them more readable
occ_dummies.columns = ['occ1', 'occ2', 'occ3', 'occ4', 'occ5', 'occ6']
hus_occ_dummies.columns = ['hocc1', 'hocc2', 'hocc3', 'hocc4', 'hocc5', 'hocc6']

# Create X and Y datasets from logistic regression
X = df.drop(labels = ['occupation', 'occupation_husb', 'Had_Affair'],
	   axis = 1) # Drop the columns 
X = pd.concat(objs = [X, occ_dummies, hus_occ_dummies], axis = 1)
print X.head()

Y = df['Had_Affair'].copy()
print '\n'
print Y.head()

# The dummy variables created are highly correlated, so to avoid
# multi-collinearlity, we drop one of the dummy columns

X = X.drop(labels = ['occ1', 'hocc1'], axis = 1)
X = X.drop(labels = ['affairs'], axis = 1) # this col is not needed anymore
print X.head()

# Flatten the Y array to 1D array, so we can use in logistic regression with sci-kit
Y = np.ravel(Y)
print Y

# All the data is now prepped for logistic regression
# Start doing the model

log_model = LogisticRegression()
log_model.fit(X = X, y = Y)
print 'Accuracy: {}'.format(log_model.score(X, Y))

# Check the coefficient dataframe
coeff_df = DataFrame(zip(X.columns, np.transpose(log_model.coef_)))
print coeff_df

# Separate the data into training and testing data set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
print 'X_train: {}, X_test: {}, Y_train: {}, Y_test: {}'.format(X_train.shape,
							       X_test.shape,
							       Y_train.shape,
							       Y_test.shape)

log_model2 = LogisticRegression()
log_model2.fit(X = X_train, y = Y_train)

class_predict = log_model2.predict(X = X_test)
# Check accuracy (This method only works with logistic regression / classification outputs
# These don't work with Linear Regression as the outputs are continous
accuracy = int(metrics.accuracy_score(y_true = Y_test, y_pred = class_predict) * 100)
print 'Prediction Accuracy: {}%'.format(accuracy)

## How can we improve our accuracy - By using Regularization (Read more about this)