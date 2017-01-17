from __future__ import division

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import matplotlib as mtp
mtp.use('TkAgg')

import matplotlib.pyplot as plt
import seaborn as sns
import scipy as stats

sns.set_style('whitegrid')

from datetime import datetime
import time

donor_df = pd.read_csv('election-donor-data.csv')
print donor_df.head()

print donor_df['contb_receipt_amt'].value_counts()
don_mean = donor_df['contb_receipt_amt'].mean()
don_std = donor_df['contb_receipt_amt'].std()

print 'Average Donation: {}, Std: {}'.format(round(don_mean, 2),
					    round(don_std), 2)
# We get a huge standard deviation, so let's see some large donations

top_donors = donor_df['contb_receipt_amt'].copy()
top_donors.sort()
print top_donors 

# Get rid of the negative amounts, which are refunds
top_donors = top_donors[top_donors > 0]
top_donors.sort()
print top_donors.value_counts().head(10)

common_donations = top_donors[top_donors < 2500]
common_donations.hist(bins = 100)
plt.show()

# Separate donations by party
start_time = time.time()
candidates = donor_df['cand_nm'].unique() # unique candidates in the dataset
candidate_vs_party_map = {}

# All the candidates are Republications except Barack Obama
for candidate_name in candidates:
    party = 'Republican'
    if candidate_name == 'Obama, Barack':
	party = 'Democrat'
    candidate_vs_party_map['{}'.format(candidate_name)] = party
print candidate_vs_party_map

donor_df['Party'] = donor_df['cand_nm'].map(candidate_vs_party_map)
print donor_df[donor_df['Party'] == 'Democrat'].head()
end_time = time.time()

print 'Time Taken by Map method: {}'.format(end_time - start_time)
print 'Number of Rows: {}'.format(len(donor_df))

donor_df = donor_df[donor_df['contb_receipt_amt'] > 0]
print donor_df.head()

print 'Number of Rows: {}'.format(len(donor_df))

# plot the total number of contributions to each candidate
num_contrib_per_cand = donor_df.groupby('cand_nm')['contb_receipt_amt'].count()
print 'Total number of contributions to each candidate'
print num_contrib_per_cand
num_contrib_per_cand.plot(kind = 'bar')
plt.suptitle('No. of contributions per candidate')
plt.show()

# plot the total contribution to each candidate
print 'Total Contribution to each candidate'
cont_per_cand = donor_df.groupby('cand_nm')['contb_receipt_amt'].sum()
print cont_per_cand

index = 0
for donation in cont_per_cand:
    print 'Candidate {} raised ${}'.format(cont_per_cand.index[index], int(round(donation, 2)))
    index = index + 1
cont_per_cand.plot(kind = 'bar')
plt.suptitle('Total contribution per candidate')
plt.show()

# plot the total number of contributions to each party
print 'Total number of contribution per party'
num_cont_per_party = donor_df.groupby('Party')['contb_receipt_amt'].count()
print num_cont_per_party
num_cont_per_party.plot(kind = 'bar')
plt.suptitle('No. of contributions per party')
plt.show()

# plot the total contribution to each party
print 'Total contribution to each Party'
cont_per_party = donor_df.groupby('Party')['contb_receipt_amt'].sum()
print cont_per_party
cont_per_party.plot(kind = 'bar')
plt.suptitle('Total contribution per party')
plt.show()

# Donations came from which occupation
occupation_df = donor_df.pivot_table(values = ['contb_receipt_amt'],
		    index = ['contbr_occupation'], columns = ['Party'],
		    aggfunc = 'sum')
print occupation_df.shape
# Restrict to occupations which in total contributed more than $1M
occupation_df = occupation_df[(occupation_df[occupation_df.columns[0]] + \
			      occupation_df[occupation_df.columns[1]]) > 1000000]
print occupation_df

# Drop Information Requested
occupation_df = occupation_df.drop(labels = ['INFORMATION REQUESTED'], axis = 0)
occupation_df.loc['CEO'] = occupation_df.loc['CEO'] + occupation_df.loc['C.E.O.']
occupation_df = occupation_df.drop(labels = ['C.E.O.'], axis = 0)

occupation_df.plot(kind = 'barh', figsize = (10, 12),
		  cmap = 'seismic')
plt.show()