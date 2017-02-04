import pandas as pd
import numpy as np
from sklearn.svm import SVC
import csv

train_df = pd.read_csv('train.csv')
cols = []
for feature_num in range(0, len(train_df.columns)):
    cols.append('feat_{}'.format(feature_num + 1))
train_df = pd.read_csv('train.csv', header = None, names = cols)
train_labels_df = pd.read_csv('trainLabels.csv', header = None,
			     names = ['target'])
test_df = pd.read_csv('test.csv', header = None, names = train_df.columns)

train_df['target'] = train_labels_df['target']

X_train = train_df[[col for col in train_df.columns if col not in ['target', 'feat_29']]]
Y_train = train_df[['target']]
Y_train = np.ravel(Y_train)

X_test = test_df[[col for col in test_df.columns if col not in ['feat_29']]]

model = SVC()
model.fit(X = X_train, y = Y_train)
print 'Fit Score: {}'.format(model.score(X_train, Y_train))

predictions = model.predict(X_test)
print predictions.shape

print 'Creating Submission File'
with open('submission.csv', 'w') as out_file:
    writer = csv.writer(out_file, delimiter = ',',
		       quotechar = '"', quoting = csv.QUOTE_NONE)
    writer.writerow(['Id', 'Solution'])
    index = 1
    for p in predictions:
	writer.writerow(['{}'.format(index), '{}'.format(p)])
	index = index + 1
print 'Finished!'