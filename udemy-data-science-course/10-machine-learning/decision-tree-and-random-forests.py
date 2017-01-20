import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib as mtp
mtp.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

import sklearn
from sklearn.datasets import make_blobs # for making a dataset to work on
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

#X, y = make_blobs(n_samples = 500, centers = 4, random_state = 8, 
#		 cluster_std = 0.1) # low cluster_std (Very much clustered)
#X, y = make_blobs(n_samples = 500, centers = 4, random_state = 8, 
#		 cluster_std = 10) # high cluster_std (Not at all clustered)
X, y = make_blobs(n_samples = 500, centers = 4, random_state = 8, 
		 cluster_std = 2.4)

# Make a scatter plot of the data
plt.figure(figsize = (6,6))
plt.scatter(X[:, 0], X[:, 1], c = y, s = 50, cmap = 'jet') # color is the Y label
plt.show()

## Make a function to visualize the classifier
def visualize_tree(classifier, X, y, boundaries = True, xlim = None, ylim = None, title = 'Decision Tree Classifier'):
    classifier.fit(X, y) # fit the X and y to the classifier model

    # set the x-limit and y-limit +/- 0.1
    if not xlim:
	xlim = (X[:, 0].min() - 0.1, X[:, 0].max() + 0.1)
    if not ylim:
	ylim = (X[:, 1].min() - 0.1, X[:, 1].max() + 0.1)
    # assign the xlim and ylim variables
    x_min, x_max = xlim
    y_min, y_max = ylim

    # create a mesh grid
    xx, yy = np.meshgrid(np.linspace(start = x_min, stop = x_max, num = 100),
			np.linspace(start = y_min, stop = y_max, num = 100))

    # define the Z by predictions (this will color in the mesh grid)
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

    # reshape based on meshgrid
    Z = Z.reshape(xx.shape)

    # plot the figure
    plt.figure(figsize = (10, 10))
    plt.pcolormesh(xx, yy, Z, alpha = 0.2, cmap = 'jet')

    # plot the training points as well
    plt.scatter(X[:, 0], X[:, 1], c = y, s = 50, cmap = 'jet')

    # set limits
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    def plot_boundaries(i, xlim, ylim):
	## This function plots the decision boundaries
	if i < 0: return

	tree = classifier.tree_

	# Recursively go through nodes of tree to plot boundaries
	if tree.feature[i] == 0:
	    plt.plot([tree.threshold[i], tree.threshold[i]], ylim, '-k')
	    plot_boundaries(tree.children_left[i], [xlim[0], tree.threshold[i]], ylim)
	    plot_boundaries(tree.children_right[i], [tree.threshold[i], xlim[1]], ylim)
	
	elif tree.feature[i] == 1:
	    plt.plot(xlim, [tree.threshold[i], tree.threshold[i]], '-k')
	    plot_boundaries(tree.children_left[i], xlim,
		    [ylim[0], tree.threshold[i]])
	    plot_boundaries(tree.children_right[i], xlim,
		    [tree.threshold[i], ylim[1]])
	
    # Random Forest vs Single Tree
    if boundaries:
	plot_boundaries(0, plt.xlim(), plt.ylim())
    
    plt.title(title)
    plt.show()

for depth in range(1, 6):
    clf = DecisionTreeClassifier(max_depth = depth, random_state = 0)
    visualize_tree(clf, X, y, title = 'Decision Tree Classifier Depth: {}'.format(depth))

depth = 100
clf = DecisionTreeClassifier(max_depth = depth, random_state = 0)
visualize_tree(clf, X, y, title = 'Decision Tree Classifier Depth: {}'.format(depth))

# With increasing depth, we will be overfitting 
# Which means that, with a new point we will be taking into account more noise than signal
# So, in the above case we will have to find an optimal depth (Ensemble Method)

## One good Ensemble Method is Random Forest

clf = RandomForestClassifier(n_estimators = 100, random_state = 0)
visualize_tree(clf, X, y, boundaries = False, title = 'Random Forest Classifier')

# Compare the accuracy of Decision Tree and Random Forest
X_train, X_test, y_train, y_test = train_test_split(X, y)
print 'X_train: {}, X_test: {}, y_train: {}, y_test: {}'.format(X_train.shape,
							       X_test.shape,
							       y_train.shape,
							       y_test.shape)

def find_accuracy_score(clf, name):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = round(metrics.accuracy_score(y_true = y_test, y_pred = y_pred) * 100, 2)
    return '{name} : {score}%'.format(name = name, score = accuracy)

print find_accuracy_score(clf = DecisionTreeClassifier(max_depth = 3, random_state = 0),
			  name = 'Decision Tree')
print find_accuracy_score(clf = RandomForestClassifier(n_estimators = 100, random_state = 0),
			 name = 'Random Forest')

## Random Forest Regression (Just like classification above)
x = 10 * np.random.rand(100)
print x

# Function to make a random sinosoidial curve for `x`
def sin_model(x, sigma = 0.2):
    noise = sigma * np.random.rand(len(x))
    return np.sin(5 * x) + np.sin(0.5 * x) + noise

y = sin_model(x)
# Plot y
plt.figure(figsize = (10, 8))
plt.errorbar(x = x, y = y, yerr = 0.2, fmt = 'o')
plt.show()

xfit = np.linspace(start = 0, stop = 10, num = 1000)

rfr = RandomForestRegressor(n_estimators = 100)
rfr.fit(x[:, None], y)
yfit = rfr.predict(xfit[:, None])
ytrue = sin_model(xfit, 0)

plt.figure(figsize = (16, 8))
plt.errorbar(x, y, 0.1, fmt = 'o')
plt.plot(xfit, yfit, '-r')
plt.plot(xfit, ytrue, '-k', alpha = 0.5)
plt.show()