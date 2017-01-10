import numpy as np
import pandas as pd

from pandas import Series, DataFrame

import requests
from bs4 import BeautifulSoup

website = 'http://en.wikipedia.org/wiki/NFL_win-loss_records'

# Read data from the HTML page
r = requests.get(url = website)
if r.ok:
    soup = BeautifulSoup(r.content)
    table = soup.find_all('table', {'class' : 'wikitable', 'class' : 'sortable'})[0]
    headers = []
    data = []
    for th in table.find_all('th'):
	headers.append(th.text)
    tr_num = 0
    for tr in table.find_all('tr'):
	if tr_num == 0:
	    # this is the header row - ignore
	    tr_num = tr_num + 1
	    continue
	row = {}
	index = 0
	for td in tr.find_all('td'):
	    row[headers[index]] = td.text
	    index = index + 1
	data.append(row)

df = DataFrame(data)
# Re-order the columns
df = DataFrame(df, columns = ['Rank', 'Team', 'Won', 'Lost', 'Tied', 'Pct.', 
			     'First NFL Season', 'Total Games', 'Division'])
print '{}'.format(df)
print 'Columns: {}'.format(df.columns)
print '--- Teams ---'
print '{}'.format(df['Team'])

print '--- Grab multiple columns from a DataFrame ---'
print '{}'.format(DataFrame(df, columns = ['Team', 'First NFL Season', 'Total Games']))

print '--- Grab columns that are not present ---'
print '{}'.format(DataFrame(df, columns = ['Team', 'First NFL Season', 'Total Games', 'Stadium']))

print '--- Get first rows ---'
print '{}'.format(df.head())

print '--- Get end rows ---'
print '{}'.format(df.tail())

print '{}'.format(df.ix[3])

df['Stadium'] = "Levi's Stadium"
print '{}'.format(df)

# Adding a Series to a DataFrame
stadiums = Series(['Levis Stadium', 'AT&T Stadium'], index = [4, 0])
print '{}'.format(stadiums)
df['Stadium'] = stadiums # assign the stadium to corresponding indices
print '{}'.format(df)

del df['Stadium']
print '{}'.format(df)

data = {'City' : ['SF', 'LA', 'NYC'], 'Population' : [83700, 888000, 34590]}
city_df = DataFrame(data)
print '{}'.format(city_df)