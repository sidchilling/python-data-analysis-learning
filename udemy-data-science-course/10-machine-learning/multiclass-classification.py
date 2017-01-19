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
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X = iris.data # features
Y = iris.target # target

print iris.DESCR

iris_data = DataFrame(X, columns = ['Sepal Length', 'Sepal Width',
				   'Petal Length', 'Petal Width'])
print iris_data.head()

iris_target = DataFrame(Y, columns = ['Species']) # species are 0, 1, and 2
print iris_target.head()
print iris_target.tail()

# Rename the species from numbers to species names
def flower(num):
    if num == 0: return 'Setosa'
    elif num == 1: return 'Versicolor'
    else:
	return 'Virginica'
iris_target['Species'] = iris_target['Species'].apply(flower)
print iris_target

iris = pd.concat(objs = [iris_data, iris_target], axis = 1)
print iris.head()
print iris.tail()

# Visulasize the data
sns.pairplot(data = iris, hue = 'Species', size = 2)
plt.show()

# Visualize Petal length distribution
sns.factorplot('Petal Length', data = iris, hue = 'Species', size = 10, kind = 'count')
plt.show()

sns.factorplot('Petal Width', data = iris, hue = 'Species', size = 10, kind = 'count')
plt.show()

sns.factorplot('Sepal Length', data = iris, hue = 'Species', size = 10, kind = 'count')
plt.show()

sns.factorplot('Sepal Width', data = iris, hue = 'Species', size = 10, kind = 'count')
plt.show()

# Create Logistic Regression
logreg = LogisticRegression()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
						   test_size = 0.4,
						   random_state = 3)
print 'X_train Shape: {}, X_test Shape: {}, Y_train Shape: {}, Y_test Shape: {}'.format(X_train.shape,
										       X_test.shape,
										       Y_train.shape,
										       Y_test.shape)
logreg.fit(X = X_train, y = Y_train)
Y_pred = logreg.predict(X = X_test)
accuracy = int(metrics.accuracy_score(y_true = Y_test, y_pred = Y_pred) * 100)
print 'Accuracy: {}%'.format(accuracy)

# K-neighbor algorithm
def run_kneighbors(num):
    knn = KNeighborsClassifier(n_neighbors = num)
    knn.fit(X = X_train, y = Y_train)
    Y_pred = knn.predict(X = X_test)
    return metrics.accuracy_score(y_true = Y_test, y_pred = Y_pred)

accuracies = []
for num_neighbors in range(1, 21):
    current_accuracy = run_kneighbors(num = num_neighbors)
    accuracies.append(current_accuracy)
    print '{} : {}'.format(num_neighbors, current_accuracy)

# Plot the accuracies
plt.plot(range(1, 21), accuracies)
plt.xlabel('k value')
plt.ylabel('Accuracy')
plt.show()

# There are other ways to find the optimal value for K (Read about this)