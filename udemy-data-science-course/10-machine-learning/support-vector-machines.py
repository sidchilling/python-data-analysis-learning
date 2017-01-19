import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib as mtp
mtp.use('TkAgg')
import matplotlib.pyplot as plt

import sklearn
from sklearn import datasets
from sklearn.svm import SVC # Support Vector Classification
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm

iris = datasets.load_iris()
X = iris.data # features
Y = iris.target # target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3,
						   random_state = 3)
print 'X_train shape: {}, X_test shape: {}, Y_train shape: {}, Y_test shape: {}'.format(X_train.shape,
										       X_test.shape,
										       Y_train.shape,
										       Y_test.shape)

model = SVC().fit(X_train, Y_train)
#model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
accuracy = round(metrics.accuracy_score(Y_test, Y_pred) * 100, 2)
print 'Accuracy: {}%'.format(accuracy)

# Trying different kernel types
X = iris.data[:, :2] # only the first two features
Y = iris.target

C = 1.0 # SVM Regularization Parameter

svc = svm.SVC(kernel = 'linear', C = C).fit(X, Y) # kernels - linear, poly, rbg, sigmoid, precomputed
rbf_svc = svm.SVC(kernel = 'rbf', gamma = 0.7, C = C).fit(X, Y)
poly_svc = svm.SVC(kernel = 'poly', degree = 3, C = C).fit(X, Y)
lin_svc = svm.LinearSVC(C = C).fit(X, Y)

# Plot 

h = 0.02 # Step-size
x_min = X[:, 0].min() - 1
x_max = X[:, 0].max() + 1

y_min = X[:, 1].min() - 1
y_max = X[:, 1].max() + 1

# Create a Mesh Grid
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
titles = [
    'SVC with Linear Kernel',
    'Linear SVC (Linear Kernel)',
    'SVC with RBF Kernel',
    'SVC with Polynomial (Degree 3) Kernel'
]

for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    # Plot the decision boundary
    plt.figure(figsize = (15, 15))
    plt.subplot(2, 2, i + 1) # 4 figures, position of the subplot, which is enumerate index + 1
    plt.subplots_adjust(wspace = 0.4, hspace = 0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap = plt.cm.terrain, alpha = 0.5, linewidths = 0)
    plt.scatter(X[:, 0], X[:, 1], c = Y, cmap = plt.cm.Dark2)

    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])
    plt.show()