# Naive Bayes Classification is for Multi-Class Classification

import pandas as pd
from pandas import Series, DataFrame

import matplotlib as mtp
mtp.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

X = iris.data
Y = iris.target

print 'X Shape: {}, Y Shape: {}'.format(X.shape, Y.shape)

accuracies = []

# Do the model 100 times
for i in range(100):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    model = GaussianNB()
    model.fit(X = X_train, y = Y_train)

    Y_predicted = model.predict(X = X_test)
    accuracy_score = round(metrics.accuracy_score(y_true = Y_test, y_pred = Y_predicted) * 100, 2)
    accuracies.append(accuracy_score)
    print 'Index: {index}, Accuracy: {score}%'.format(index = i, score = accuracy_score)

# Let's plot the accuracies
df = DataFrame({'score' : accuracies})
plt.plot(df)
plt.show()