import pandas as pd
from pandas import DataFrame
import matplotlib as mtp
mtp.use('TkAgg')
import matplotlib.pyplot as plt

import large_number_fmt as fmter
import time
import random
import operator

import ml_metrics
from sklearn.decomposition import PCA
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', None)

destination_df = pd.read_csv('destinations.csv')

# The files are pretty big to read completely, so just reading a subset
test_df = pd.read_csv('test.csv') # Read all rows
train_df = pd.read_csv('train.csv', nrows = 10000000) # Read 1Cr rows 

print 'Destinations: Rows = {}, Cols = {}'.format(fmter.fmt(destination_df.shape[0]),
						 fmter.fmt(destination_df.shape[1]))
print 'Test: Rows = {}, Cols = {}'.format(fmter.fmt(test_df.shape[0]),
					 fmter.fmt(test_df.shape[1]))
print 'Train: Rows = {}, Cols = {}'.format(fmter.fmt(train_df.shape[0]),
					  fmter.fmt(train_df.shape[1]))

print train_df.head(n = 10) # Display 10 rows
print '-------------'
print test_df.head(n = 10)

cols = train_df.columns
cols = list(set(cols) - set(['date_time', 'orig_destinatation_distance', 
		       'srch_ci', 'srch_co', 'srch_adults_cnt',
		       'srch_children_cnt', 'srch_rm_cnt',
		       'srch_destination_id', 'srch_destination_type_id',
		       'cnt']))
# Find the unique values for each col
for col in cols:
    print '# of unique values for {} = {}'.format(col, len(train_df[col].unique()))

# Find the cols that are present in train_df but not in test_df
diff = list(set(train_df.columns) - set(test_df.columns))
print 'Cols not present in test_df: {}'.format(diff)

# We have to predict which hotel_cluster the user will book. There are
# 100 hotel clusters

# Now, derive a sample training and testing dataframe from `train`
# so that we can apply prediction techniques and check for correctness.
# In the description of the problem, it says that the test dataset has 
# dates that are ahead of the dates in the train dataframe. So, for
# dividing the train dataset, we can divide by time

# Convert date_time to datetime object and add month and year column
train_df['date_time'] = pd.to_datetime(train_df['date_time'])
train_df['month'] = train_df['date_time'].dt.month
train_df['year'] = train_df['date_time'].dt.year
print train_df.head()

# We have preserve data from one user. We select random 10K users
# and select from train data for those 10K users
unique_users = train_df['user_id'].unique()
sel_user_ids = [unique_users[i] for i in sorted(random.sample(range(len(unique_users)), 10000))]
sel_train = train_df[train_df['user_id'].isin(sel_user_ids)]

# `sel_train` contains all the data for the `sel_user_ids`

# Create a new training and testing dataset by dividing based on time
# Anything below July 2014 will be training data and above that will be testing data
t1 = sel_train[(sel_train['year'] == 2013) | ((sel_train['year'] == 2014) & \
					     (sel_train['month'] < 8))] # training data
t2 = sel_train[((sel_train['year'] == 2014) & (sel_train['month'] >= 8)) | \
	      (sel_train['year'] > 2014)] # testing data

# As the final test data contains only booking events, `t2` should also contain
# data for booking events
t2 = t2[t2['is_booking'] == 1]

print 'Training Data #rows = {}, Testing Data #rows = {}'.format(t1.shape[0], t2.shape[0])

# Now, we will start with a simple prediction
# One simple prediction could be to find the 5 most common hotel clusters
# from train data, and predict that for each row in `t2`, the result is the
# 5 most common hotel clusters
common_clusters = list(train_df['hotel_cluster'].value_counts().head().index)
print '5 Most common hotel clusters = {}'.format(common_clusters)

# For each row of t2, we say the prediction is `common_clusters`
predictions = [common_clusters for i in range(0, t2.shape[0])]

# Now, calculate accuracy of the predictions
target = [[l] for l in t2['hotel_cluster']]
print 'Accuracy = {}%'.format(round(ml_metrics.mapk(actual = target,
					    predicted = predictions, k = 5) * 100), 2)

# Let's dive into correlation. We will check if any colums correlate well
# with our hotel_cluster
print train_df.corr()['hotel_cluster']

# From the above correlation, we see that there no such high linear
# correlation between columns and hotel_cluster.
# This means that we cannot use logistic or linear regression models as they
# rely on having a linear correlation

# So, let's start with generating features
# Let's start with generating features from destination_df
# destination_df contains 149 cols that will make any machine learning algorithm
# very slow. So, we use PCA to compress the 149 cols to 3 cols while preserving
# the variance
print 'Shape of Destination Dataset = {}'.format(destination_df.shape)
pca = PCA(n_components = 3) # 3 columns
dest_small_df = pca.fit_transform(destination_df[['d{}'.format(i + 1) for i in range(0, destination_df.shape[1] - 1)]])
dest_small_df = DataFrame(dest_small_df)
dest_small_df['srch_destination_id'] = destination_df['srch_destination_id']
print dest_small_df.head()

# Now we will add features.
# Date features like month, day, hour, minute, dayofweek, quarter
# The above date features for date_time, srch_ci, srch_co
# We will add stay_span, as time duration of the stay

df = t1.copy() # make a copy to change that copy
df['date_time'] = pd.to_datetime(df['date_time'])
df['srch_ci'] = pd.to_datetime(df['srch_ci'], format = '%Y-%m-%d',
			      errors = 'coerce')
df['srch_co'] = pd.to_datetime(df['srch_co'], format = '%Y-%m-%d',
			      errors = 'coerce')

date_props = ['month', 'day', 'hour', 'minute', 'dayofweek', 'quarter']
for date_prop in date_props:
    df['{}'.format(date_prop)] = getattr(df['date_time'].dt, date_prop)
date_props = ['month', 'day', 'dayofweek', 'quarter']
for date_prop in date_props:
    df['ci_{}'.format(date_prop)] = getattr(df['srch_ci'].dt, date_prop)
    df['co_{}'.format(date_prop)] = getattr(df['srch_co'].dt, date_prop)
df['stay_span'] = (df['srch_co'] - df['srch_ci']).astype('timedelta64[h]')
df = df.drop(labels = ['date_time', 'srch_ci', 'srch_co'], axis = 1)
df = df.join(other = dest_small_df, on = 'srch_destination_id', how = 'left',
	    rsuffix = 'dest')
df = df.drop(labels = ['srch_destination_iddest'], axis = 1)
df = df.fillna(-1)
print df.head()

# Let's see if Random Forest gives us what accuracy? (Hunch is that 
# machine learning will not predict good results as there is no
# correlation between the columsns and hotel_cluster
predictors = [c for c in df.columns if c not in ['hotel_cluster']]
clf = RandomForestClassifier(n_estimators = 10, min_weight_fraction_leaf = 0.1)
scores = model_selection.cross_val_score(estimator = clf,
					X = df[predictors], 
					y = df['hotel_cluster'],
					cv = 3)
scores = ['{}%'.format(round((score * 100), 2)) for score in scores]
print scores

# The above scores are pretty bad. This confirms our hunch that machine learning
# will not work as there are over 100 hotel clusters and no correlation
# between columns and our target column

def make_key(items):
    return '-'.join(['{}'.format(i) for i in items])

match_cols = ['srch_destination_id']
cluster_cols = match_cols + ['hotel_cluster']

groups = t1.groupby(cluster_cols)
top_clusters = {}
for name, group in groups:
    clicks = len(group[group['is_booking'] == False])
    bookings = len(group[group['is_booking'] == True])

    score = bookings + (0.15 * clicks)

    clus_name = make_key(name[:len(match_cols)])
    if not top_clusters.get(clus_name, None):
	top_clusters[clus_name] = {}
    top_clusters[clus_name][name[-1]] = score

print top_clusters

cluster_dict = {}
for n, tc in top_clusters.iteritems():
    top = [l[0] for l in sorted(tc.items(), key = operator.itemgetter(1),
				reverse = True)[:5]]
    cluster_dict[n] = top

print cluster_dict

preds = []
for index, row in t2.iterrows():
    key = make_key([row[m] for m in match_cols])
    preds.append(cluster_dict.get(key, []))
print preds

target = [[l] for l in t2['hotel_cluster']]
print 'Accuracy = {}%'.format(round(ml_metrics.mapk(actual = target,
					    predicted = preds, k = 5) * 100), 2)

match_cols = ['user_location_country', 
	     'user_location_region',
	     'user_location_city',
	     'hotel_market',
	     'orig_destination_distance']

groups = t1.groupby(match_cols)

def generate_exact_matches(row, match_cols):
    index = tuple([row[t] for t in match_cols])
    try:
	group = groups.get_group(index)
    except:
	return []
    clus = list(set(group.hotel_cluster))
    return clus

exact_matches = []
for i in range(t2.shape[0]):
    exact_matches.append(generate_exact_matches(t2.iloc[i], match_cols))

def f5(seq, idfun = None):
    if not idfun:
	def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
	marker = idfun(item)
	if marker in seen: continue
	seen[marker] = 1
	result.append(item)
    return result

full_preds = [f5(exact_matches[p] + preds[p] + common_clusters)[:5] for p in range(len(preds))]
print 'Accuracy: {}%'.format(round((ml_metrics.mapk([[l] for l in t2['hotel_cluster']], full_preds, k = 5) * 100), 2))

# Make the submission files
write_p = [' '.join(['{}'.format(l) for l in p]) for p in full_preds]
write_frame = ['{}{}'.format(t2['id'][i], write_p[i]) for i in range(len(full_preds))]
write_frame = ['id, hotel_clusters'] + write_frame
with open('predictions.csv', 'w+') as f:
    f.write('\n'.join(write_frame))

print 'Finished!'