import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import matplotlib as mtp
mtp.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

import string
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

messages = [line.rstrip() for line in open('sms-spam-collection')]
print len(messages)

for num, message in enumerate(messages[:10]):
    print '{} : {}'.format(num, message)

messages = pd.read_csv('sms-spam-collection', sep = '\t', 
		      names = ['label', 'message'])
print messages.head()
print messages.describe()
print messages.groupby('label').describe()

## Read about feature engineering (from the jupyter notebook)
messages['length'] = messages['message'].apply(len)
print messages.head(10)

# Visualize some data
messages['length'].hist(bins = 30)
plt.show()

# There must be some messages that are really long
d = messages['length'].describe()
print d
print messages[messages['length'] == int(d['max'])].iloc[0]['message']

messages.hist(column = 'length', by = 'label', bins = 50, figsize = (10, 4))
plt.show()

## Machine learning classification requires some numerical data
# We will convert the text data into numerical data

# Bag of words - In this approach, every word is represented by a number
## Function to process the message
def text_process(message):
    # Remove puncutations
    nopunc = ''.join([char for char in message if char not in string.punctuation])
    # Remove stop words
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

print messages['message'].head(5).apply(text_process)

# We can do a lot of normalization of text (Read more about this)

## Vectorization

msg_train, msg_test, label_train, label_test = train_test_split(messages['message'],
							       messages['label'],
							       test_size = 0.2)

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer = text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])
pipeline.fit(msg_train, label_train)

predictions = pipeline.predict(msg_test)
print classification_report(predictions, label_test)