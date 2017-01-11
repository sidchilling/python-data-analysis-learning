import numpy as np
import pandas as pd

import matplotlib as mtp
mtp.use('TkAgg')

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

flights_df = sns.load_dataset('flights')
print flights_df.head()

months_map = {
    'January' : 'Jan',
    'February' : 'Feb',
    'March' : 'Mar',
    'April' : 'Apr',
    'May' : 'May',
    'June' : 'Jun',
    'July' : 'Jul',
    'August' : 'Aug',
    'September' : 'Sep',
    'October' : 'Oct',
    'November' : 'Nov',
    'December' : 'Dec'
}
flights_df['month'] = flights_df['month'].map(months_map)
print flights_df.head()

flights_df = flights_df.pivot('month', 'year', 'passengers')
print flights_df.head()

# make a heatmap from the above data
sns.heatmap(data = flights_df, annot = True, fmt = 'd')
plt.show()

# specify a center for the heatmap color
sns.heatmap(data = flights_df, annot = True, fmt = 'd',
	   center = flights_df.loc['Jan', 1955]) # basically the center of the dataframe
plt.show()

# subplots 
f, (axis1, axis2) = plt.subplots(2, 1) # 2 rows and 1 column in the full plot

yearly_flights = flights_df.sum()
years = pd.Series(yearly_flights.index.values)
years = pd.DataFrame(years)
print years
print '\n\n'

flights = pd.Series(yearly_flights.values)
flights = pd.DataFrame(flights)
print flights
print '\n\n'

year_df = pd.concat((years, flights), axis = 1)
year_df.columns = ['Year', 'Flights']
print year_df
print '\n\n'

sns.barplot(x = 'Year', y = 'Flights', data = year_df, ax = axis1)
sns.heatmap(data = flights_df, cmap = 'Blues', ax = axis2,
	    cbar_kws = {'orientation' : 'horizontal'})
plt.show()

# Cluster Map
sns.clustermap(data = flights_df)
plt.show()

sns.clustermap(data = flights_df, col_cluster = False) # only cluster by rows
plt.show()

# setting a standard scale
sns.clustermap(data = flights_df, standard_scale = 1) # scale by cols (months)
plt.show()

sns.clustermap(data = flights_df, standard_scale = 0)
plt.show()

# z-score
sns.clustermap(data = flights_df, z_score = 1)
plt.show()
