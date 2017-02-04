import pandas as pd
import numpy as np

import matplotlib as mtp
mtp.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import metrics

from operator import itemgetter

pd.set_option('display.max_columns', None)

train_df = pd.read_csv('train.csv')
cols = []
for feature_num in range(0, len(train_df.columns)):
    cols.append('feat_{}'.format(feature_num + 1))
train_df = pd.read_csv('train.csv', header = None, names = cols)

train_labels_df = pd.read_csv('trainLabels.csv', header = None,
			     names = ['target'])
test_df = pd.read_csv('test.csv', header = None, names = train_df.columns)

train_df['target'] = train_labels_df['target']

print train_df.head()
print test_df.head()
print train_df['target'].unique() # [0, 1]

# As there are only two possible values for the target, it's a classification 
# problem

# There are 40 features (looks like a large number), we can try taking into 
# consideration all the features or we can reduce the number of features before
# applying machine learning classification algorithms. We will try both approaches
# to see which ones give us better accuracy

# Let's first see if there is any correlation between the features
corr_df = train_df[[col for col in train_df.columns if col != 'target']].corr()
sns.linearmodels.corrplot(corr_df, annot = False, diag_names = False)
plt.show() # Uncomment to show the graph

# From the above graph, we see that there might be a high correlation 
# between (feat_5, [feat_23, feat_24]) and (feat_13, [feat_29]). So, let's see
# what that correlation exactly is.
print train_df[['feat_5', 'feat_23', 'feat_24']].corr()
print train_df[['feat_13', 'feat_29']].corr()

# From the above values, we can conclude that feat_13 and feat_29 are 
# highly correlated (but not the others). We will use this later

# Let's use different classification learning algorithms on the whole 
# data now and check for accuracy

def predict(X_train, y_train, X_test, y_test, model):
    model.fit(X = X_train, y = y_train)
    predictions = model.predict(X = X_test)
    fit_accuracy = model.score(X_train, y_train)
    accuracy = metrics.accuracy_score(y_true = y_test,
				     y_pred = predictions)
    return (fit_accuracy, accuracy)

def run_all_algorithms(X, Y):
    models_vs_accuracies = [
	{'name' : 'LogisticRegression', 'model' : LogisticRegression(), 'accuracies' : [], 'avg_pred_accuracy' : None},
	{'name' : 'DecisionTreeClassifier', 'model' : DecisionTreeClassifier(), 'accuracies' : [], 'avg_pred_accuracy' : None},
	{'name' : 'SupportVectorMachines', 'model' : SVC(), 'accuracies': [], 'avg_pred_accuracy' : None},
	{'name' : 'NaiveBayes-Gaussian', 'model' : GaussianNB(), 'accuracies' : [], 'avg_pred_accuracy' : None},
	{'name' : 'KNN', 'model' : KNeighborsClassifier(n_neighbors = 10), 'accuracies' : [], 'avg_pred_accuracy' : None}
    ]

    for m in models_vs_accuracies:
	for i in range(0, 10):
	    X_train, X_test, y_train, y_test = train_test_split(X, Y)
	    # LogisticRegression
	    fit_accuracy, prediction_accuracy = predict(X_train = X_train,
							y_train = y_train,
							X_test = X_test,
							y_test = y_test,
							model = m['model'])
	    m['accuracies'].append({
		'fit' : fit_accuracy,
		'prediction' : prediction_accuracy
	    })

    for m in models_vs_accuracies:
	fit_accuracy = np.average([x['fit'] for x in m['accuracies']])
	pred_accuracy = np.average([x['prediction'] for x in m['accuracies']])
	m['avg_pred_accuracy'] = pred_accuracy
	print '{} Accuracy, Fit = {}%, Prediction = {}%'.format(m['name'],
							       round(fit_accuracy * 100, 2),
							       round(pred_accuracy * 100, 2))
    return [{'name' : m['name'], 'accuracy' : m['avg_pred_accuracy']} for m in models_vs_accuracies]

all_accuracies = []

def add_to_all_accuracies(name, accuracies):
    for a in accuracies:
	all_accuracies.append({
	    'name' : name,
	    'model_name' : a['name'],
	    'accuracy' : a['accuracy']
	})

# With plain X and Y
print '===================='
print 'ALL FEATURES'
print '===================='
X = train_df[[col for col in train_df.columns if col != 'target']]
Y = train_df[['target']]
Y = np.ravel(Y)
add_to_all_accuracies('All Features', run_all_algorithms(X, Y))

# Now, because feat_13 and feat_29 are highly correlated, let's just keep one
# and delete the other and run the machine learning algorithms to see
# if we get any improvement in accuracy

# With feat_13 removed
print '===================='
print 'FEAT_13 REMOVED'
print '===================='
X = train_df[[col for col in train_df.columns if col not in ['target', 'feat_13']]]
Y = train_df[['target']]
Y = np.ravel(Y)
add_to_all_accuracies('Feat_13 Removed', run_all_algorithms(X, Y))

# With feat_29 removed
print '===================='
print 'FEAT_29 REMOVED'
print '===================='
X = train_df[[col for col in train_df.columns if col not in ['target', 'feat_29']]]
Y = train_df[['target']]
Y = np.ravel(Y)
add_to_all_accuracies('Feat_29 Removed', run_all_algorithms(X, Y))

# Now, using PCA let's try to reduce the number of cols to see if
# classification accuracy increases
for num_components in range(3, 41):
    print '===================='
    print 'PCA with {} Components'.format(num_components)
    print '===================='
    X = train_df[[col for col in train_df.columns if col != 'target']]
    Y = train_df[['target']]
    Y = np.ravel(Y)
    X = PCA(n_components = num_components).fit_transform(X)
    print 'Shape: {}'.format(X.shape)
    add_to_all_accuracies('PCA {} Components'.format(num_components),
			 run_all_algorithms(X, Y))

all_accuracies = sorted(all_accuracies, key = itemgetter('accuracy'),
		       reverse = True)
print ['{}-{}-{}%'.format(x['name'], x['model_name'], round(x['accuracy'] * 100, 2)) for x in all_accuracies[:10]]

# We find that SVM performs the best, PCA-28 is the best, but the accuracies are 
# very close to each other, so we decide to drop feat_29 on the test
# and make the predictions
