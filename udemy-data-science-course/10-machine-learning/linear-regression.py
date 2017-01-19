import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import matplotlib as mtp
mtp.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from sklearn.datasets import load_boston

import sklearn
from sklearn.linear_model import LinearRegression
import sklearn.model_selection as mdlslc

boston = load_boston()
print boston.DESCR # scikit learn has given this decorator attribute

# Quick visualization of the data

plt.hist(x = boston.target, bins = 50)
plt.xlabel('Prices in $1000s')
plt.ylabel('Number of houses')
plt.show()

# Housing price vs number of rooms in the dwelling
plt.scatter(x = boston.data[:, 5], y = boston.target)
plt.ylabel('Price in $1000s')
plt.xlabel('No. of Rooms')
plt.show()

# Convert the data to the DataFrame
boston_df = DataFrame(boston.data)
boston_df.columns = boston.feature_names
boston_df['Price'] = boston.target
print boston_df.head()

sns.lmplot(x = 'RM', y = 'Price', data = boston_df)
plt.xlabel('No of Rooms')
plt.ylabel('Price in $1000s')
plt.show()

# Least Squares Method for Linear Regression (No. of Rooms to predict Price)
# Univariate Regression
X = np.vstack(boston_df['RM'])
print 'Shape of X: {}'.format(X.shape)
Y = boston_df['Price']
print 'Shape of Y: {}'.format(Y.shape)

# Transfor X to the form [X 1]
X = np.array([[value, 1] for value in X])

m, b = np.linalg.lstsq(a = X, b = Y)[0]
print 'm: {}, b: {}'.format(m, b)

plt.plot(boston_df['RM'], boston_df['Price'], 'o')
x = boston_df['RM']
plt.plot(x , (m * x) + b, 'r', label = 'Best Fit Line')
plt.show()

# Finding the error
result = np.linalg.lstsq(a = X, b = Y)
error_total = result[1]
# Root Mean Square Error
rmse = np.sqrt(error_total / len(X))
print 'The root mean square is: {}'.format(round(rmse, 3))

# Multi-variate Linear Regression
lreg = LinearRegression() # linear regression object

# Data columns (features)
X_multi = boston_df.drop(labels = ['Price'], axis = 1) # dropping the column
# The target (value)
Y_target = boston_df['Price']

# Fit a model
lreg.fit(X = X_multi, y = Y_target)
print 'The Estimated intercept coefficient is: {}'.format(round(lreg.intercept_, 2))
print 'The number of coefficients used: {}'.format(len(lreg.coef_))
print 'The coefficients are: {}'.format(lreg.coef_)

# show the coefficients for all the features
coeff_df = DataFrame(boston_df.columns)
coeff_df.columns = ['Features']
coeff_df['Coefficient Estimate'] = Series(lreg.coef_)
print coeff_df

# Predict with the same dataset
Y_compare = DataFrame({'Actual' : Y_target, 'Estimated' : lreg.predict(X = X_multi)})
print Y_compare.head(20)

# modify `Y_compare` to make it amenable to plotting
Y_compare = Y_compare.stack().reset_index()
Y_compare.columns = ['X', 'Price Type', 'Price']
print Y_compare.head()

sns.pointplot(x = 'X', y = 'Price', hue = 'Price Type',
	     data = Y_compare, linestyles = '')
plt.xlabel('')
plt.ylabel('Price')
plt.show()

## Validation

# divide into training set and testing set
X_train, X_test, Y_train, Y_test = mdlslc.train_test_split(X, boston_df['Price'])
print 'X_train: {}, X_test: {}, Y_train: {}, Y_test: {}'.format(X_train.shape,
							       X_test.shape,
							       Y_train.shape,
							       Y_test.shape)

# Make regression with the training set
lreg = LinearRegression()
lreg.fit(X = X_train, y = Y_train)

# Predict with
pred_train = lreg.predict(X = X_train)
pred_test = lreg.predict(X = X_test)

train_mse = round(np.mean((Y_train - pred_train) ** 2), 2)
print 'MSE with Y_train: {}'.format(train_mse)

pred_mse = round(np.mean((Y_test - pred_test) ** 2), 2)
print 'MSE for Y_test: {}'.format(pred_mse)

# Visualize the prediction accuracy (Residual Plots)
train = plt.scatter(x = pred_train, y = (pred_train - Y_train), 
		   c = 'b', alpha = 0.5)
test = plt.scatter(x = pred_test, y = (pred_test - Y_test),
		  c = 'r', alpha = 0.5)
plt.hlines(y = 0, xmin = -10, xmax = 40) # Have to play around here
plt.legend((train, test), ('Training', 'Test'), loc = 'lower left')
plt.title('Residual Plots')
plt.show()