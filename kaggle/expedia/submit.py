import pandas as pd
from operator import itemgetter
import random
import ml_metrics

def make_key(items):
    return '-'.join(['{}'.format(i) for i in items])

def generate_exact_matches(groups, row, match_cols):
    index = tuple([row[t] for t in match_cols])
    try:
	group = groups.get_group(index)
    except:
	return []
    clus = list(set(group.hotel_cluster))
    return clus

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

print 'Reading Data Files'
destination_df = pd.read_csv('destinations.csv')
#train_df = pd.read_csv('train.csv', nrows = 10000000) # Local
train_df = pd.read_csv('train.csv') # Production
test_df = pd.read_csv('test.csv')

'''
#For Local
train_df['date_time'] = pd.to_datetime(train_df['date_time'])
train_df['month'] = train_df['date_time'].dt.month
train_df['year'] = train_df['date_time'].dt.year

unique_users = train_df['user_id'].unique()
sel_user_ids = [unique_users[i] for i in sorted(random.sample(range(len(unique_users)), 10000))]
sel_train = train_df[train_df['user_id'].isin(sel_user_ids)]
t1 = sel_train[(sel_train['year'] == 2013) | ((sel_train['year'] == 2014) & \
					     (sel_train['month'] < 8))]
'''
t2 = test_df # For Production
t1 = train_df # For Production

print 'Computing Common Clusters'
common_clusters = list(train_df['hotel_cluster'].value_counts().head().index)

match_cols = ['srch_destination_id']
cluster_cols = match_cols + ['hotel_cluster']

groups = t1.groupby(cluster_cols)

print 'Computing Top Clusters'
top_clusters = {}
for name, group in groups:
    clicks = len(group[group['is_booking'] == False])
    bookings = len(group[group['is_booking'] == True])

    score = bookings + (0.15 * clicks)

    clus_name = make_key(name[:len(match_cols)])
    if not top_clusters.get(clus_name, None):
	top_clusters[clus_name] = {}
    top_clusters[clus_name][name[-1]] = score

print 'Computing Cluster Dict'
cluster_dict = {}
for n, tc in top_clusters.iteritems():
    top = [l[0] for l in sorted(tc.items(), key = itemgetter(1),
				reverse = True)[:5]]
    cluster_dict[n] = top

print 'Computing Preds'
preds = []
for index, row in t2.iterrows():
    key = make_key([row[m] for m in match_cols])
    preds.append(cluster_dict.get(key, []))

match_cols = ['user_location_country', 
	     'user_location_region',
	     'user_location_city',
	     'hotel_market',
	     'orig_destination_distance']
groups = t1.groupby(match_cols)

print 'Computing Exact Matches'
exact_matches = []
for i in range(t2.shape[0]):
    exact_matches.append(generate_exact_matches(groups, t2.iloc[i], match_cols))

print 'Computing Full Predictions'
full_preds = [f5(exact_matches[p] + preds[p] + common_clusters)[:5] for p in range(len(preds))]

print 'Creating the submission file'
# Make a submission file
write_p = [' '.join(['{}'.format(l) for l in p]) for p in full_preds]
write_frame = ['{}{}'.format(t2['id'][i], write_p[i]) for i in range(len(full_preds))]
write_frame = ['id, hotel_clusters'] + write_frame
with open('predictions.csv', 'w+') as f:
    f.write('\n'.join(write_frame))

