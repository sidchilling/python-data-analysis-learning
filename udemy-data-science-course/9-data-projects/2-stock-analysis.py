from __future__ import division

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

import matplotlib as mtp
mtp.use('TkAgg')

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import time

# 1. What was the change in price of the stock over time?
# 2. What was the daily return of the stock on average?
# 3. What was the moving average of the various stocks?
# 4. What was the correlation between different stocks' closing prices?
# 5. What was the correlation between different stocks' daily returns?
# 6. How much value we put at risk by investing in a particular stock?
# 7. How can we attempt to predict future stock behaviour?

sns.set_style('whitegrid') # setting the style as white

from pandas_datareader.data import DataReader
from datetime import datetime
import sys

def print_dataframes(dfs, num = 5):
    for stock in dfs.keys():
	print '--- DataFrame for {} ---'.format(stock)
	print dfs[stock].head(n = num)

def flatten_axes(axes):
    axes_array = []
    for row in range(0, len(axes)):
	for col in range(0, len(axes[row])):
	    axes_array.append(axes[row][col])
    return axes_array

def convert_date(d):
    # there might be timestamp data if we fetch the data from internet
    # there will not be timestamp data if we fetch from files
    if ' ' in d:
	d = d.split(' ')[0].strip()
    return datetime.strptime(d, '%Y-%m-%d').strftime("%d %b '%y")

tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']

end = datetime.utcnow()
start = datetime(end.year - 1, end.month, end.day)

dfs = {} # Map to store the dataframes for each stock
for stock in tech_list:
    fetch_data = False
    # first check whether this can be read from local file
    try:
	dfs[stock] = pd.read_csv('{}.csv'.format(stock))
    except:
	fetch_data = True
    if fetch_data or dfs[stock].empty:
	# fetch data from Yahoo
	print 'Fetching data for: {}'.format(stock)
	dfs[stock] = DataReader(name = stock, data_source = 'yahoo',
			 start = start, end = end)
	# save it locally
	dfs[stock].to_csv('{}.csv'.format(stock))

print_dataframes(dfs)

print dfs[tech_list[0]].describe()

# Set Date as Index 
for stock in dfs.keys():
    dfs[stock] = dfs[stock].reset_index()
    dfs[stock]['Date'] = dfs[stock]['Date'].apply(str)
    dfs[stock]['Date'] = dfs[stock]['Date'].apply(convert_date)
    dfs[stock] = dfs[stock].set_index(keys = ['Date'])
print_dataframes(dfs)

def make_subplots(rows = 2, cols = 2):
    fig, axes = plt.subplots(nrows = rows, ncols = 2)
    plt.subplots_adjust(wspace = 1, hspace = 1)
    axes_array = flatten_axes(axes)
    return (fig, axes_array)

# Historical trend of closing prices
(fig, axes_array) = make_subplots()
index = 0
for stock in tech_list:
    dfs[stock]['Adj Close'].plot(legend = True, title = '{} Adj Close Trend'.format(stock),
				ax = axes_array[index], y = 'Date')
    index = index + 1
plt.show()

# Show the Volume trend of AAPL
dfs['AAPL']['Volume'].plot(legend = True, figsize = (10, 4),
			  title = 'AAPL Volume Trend')
plt.show()

# Calculate moving average for all the stocks
ma_days = [10, 20, 50, 70, 100]
for ma in ma_days:
    column_name = 'MA for {} days'.format(ma)
    for stock in dfs.keys():
	dfs[stock][column_name] = pd.rolling_mean(arg = dfs[stock]['Adj Close'],
					   window = ma)
print_dataframes(dfs, num = 100)

def plot_moving_averages(ma_days, close = True):
    (fig, axes_array) = make_subplots()
    index = 0
    for stock in dfs.keys():
	col_names = ['Adj Close'] if close else []
	for ma in ma_days:
	    col_names.append('MA for {} days'.format(ma))
	dfs[stock][col_names].plot(legend = True, title = '{} MA'.format(stock),
				  ax = axes_array[index])
	index = index + 1
    plt.show()

# Plot the Moving averages for all the stocks for Adj Close, 10, and 20
plot_moving_averages(ma_days = [10, 20])

# Plot the moving averages for all stocks for 50, 70, 100
plot_moving_averages(ma_days = [50, 70, 100], close = False)

## Daily Returns and Risk of the Stock
for stock in dfs.keys():
    dfs[stock]['Daily Return'] = dfs[stock]['Adj Close'].pct_change()
print_dataframes(dfs)
(fig, axes_array) = make_subplots()
index = 0
for stock in dfs.keys():
    dfs[stock]['Daily Return'].plot(legend = True, title = 'Daily Return {}'.format(stock),
				   linestyle = '--', marker = 'o',
				   ax = axes_array[index])
    index = index + 1
plt.show()

# Show the daily returns on a histogram
(fig, axes_array) = make_subplots()
index = 0
for stock in dfs.keys():
    g = sns.distplot(a = dfs[stock]['Daily Return'].dropna(), bins = 100, hist = True,
		kde = True, rug = False, ax = axes_array[index])
    g.set_title('{}'.format(stock))
    index = index + 1
plt.show()

## Make a DataFrame of all the Adj Close prices for each stock
closing_dfs = DataFrame()
for stock in dfs.keys():
    adj_close = dfs[stock]['Adj Close']
    adj_close.name = '{}'.format(stock)
    closing_dfs = pd.concat([closing_dfs, adj_close], axis = 1)
print closing_dfs.head()

tech_returns = closing_dfs.pct_change()
print tech_returns.head()

# Show correlation between same stock
sns.jointplot(x = 'GOOG', y = 'GOOG', data = tech_returns, kind = 'scatter',
	     color = 'seagreen')
plt.show()

# Correlation betwenn GOOG and MSFT
sns.jointplot(x = 'GOOG', y = 'MSFT', data = tech_returns, kind = 'scatter',
	     color = 'seagreen')
plt.show()

# Show correlation between all the stocks
sns.pairplot(data = tech_returns.dropna())
plt.show()

# Show correlation using PairGrid to control the types of graphs
returns_fig = sns.PairGrid(data = tech_returns.dropna())
returns_fig.map_upper(plt.scatter, color = 'purple')
returns_fig.map_lower(sns.kdeplot, cmap = 'cool_d')
returns_fig.map_diag(plt.hist, bins = 30)
plt.show()

# Correlation between Closing prices 
returns_fig = sns.PairGrid(data = closing_dfs.dropna())
returns_fig.map_upper(plt.scatter, color = 'purple')
returns_fig.map_lower(sns.kdeplot, cmap = 'cool_d')
returns_fig.map_diag(plt.hist, bins = 30)
plt.show()

sns.linearmodels.corrplot(tech_returns.dropna(), annot = True)
plt.show()

sns.linearmodels.corrplot(closing_dfs.dropna(), annot = True)
plt.show()

## Quantify Risk

area = np.pi * 20 # so that the points that we draw are visible
plt.scatter(x = tech_returns.dropna().mean(),
	   y = tech_returns.dropna().std(),
	   s = area)
plt.xlabel('Expected Return')
plt.ylabel('Risk')

for label, x, y in zip(tech_returns.columns, tech_returns.dropna().mean(),
		       tech_returns.dropna().std()):
    plt.annotate(label, xy = (x, y), xytext = (50, 50),
		textcoords = 'offset points', ha = 'right', va = 'bottom',
		 arrowprops = {'arrowstyle' : '-', 'connectionstyle' : 'arc3,rad=-0.3'})
plt.show()

## Value at Risk

sns.distplot(a = dfs['AAPL']['Daily Return'].dropna(), bins = 100,
	   hist = True, kde = True, rug = False, color = 'purple')
plt.show()

## Bootstap Method

# Print the Quantiles
print tech_returns['AAPL'].dropna().quantile(0.05) # this means that for 95% times, this will be your worst loss
# Do the above for all the stocks
for col in tech_returns.columns:
    stock = '{}'.format(col)
    print '{}. Risk: {}'.format(stock,
			       tech_returns[stock].dropna().quantile(0.05))

## Monte-Carlo Method

def stock_monte_carlo(start_price, days, mu, sigma, dt):
    price = np.zeros(days)
    price[0] = start_price

    shock = np.zeros(days)
    drift = np.zeros(days)

    for x in xrange(1, days):
	shock[x] = np.random.normal(loc = mu * dt, scale = sigma * np.sqrt(dt))
	drift[x] = mu * dt
	price[x] = price[x - 1] + (price[x - 1] * (drift[x] + shock[x]))

    return price

# Run the Monte carlo method for Google 100 times starting with the 
# first opening price
start_price = dfs['GOOG']['Open'][0]
days = 365
dt = 1 / days
mu = tech_returns['GOOG'].mean()
sigma = tech_returns['GOOG'].std()
for run in xrange(100):
    plt.plot(stock_monte_carlo(start_price = start_price, days = days, 
			      mu = mu, sigma = sigma, dt = dt))

plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Monte Carlo Analysis for Google')
plt.show()

# Run 10,000 Monte-Carlo Simulations for all stocks
for stock in dfs.keys():
    runs = 10000
    print 'Running {} Monte-Carlo Simulations for {}'.format(runs, stock)
    simulations = np.zeros(runs)
    start_price = dfs[stock]['Open'][0]
    days = 365
    dt = 1 / days
    
    start_time = time.time()

    mu = tech_returns[stock].mean()
    sigma = tech_returns[stock].std()
    for run in xrange(runs):
	simulations[run] = stock_monte_carlo(start_price = start_price,
				      days = days, mu = mu,
				      sigma = sigma, dt = dt)[days - 1]
	# in the previous step we are taking the final day's simulated price
    q = np.percentile(simulations, 1)
    plt.hist(simulations, bins = 200)
    plt.figtext(x = 0.6, y = 0.8, s = 'Start Price: ${}'.format(round(float(start_price), 2)))
    # Mean ending Price
    plt.figtext(x = 0.6, y = 0.7,
		s = 'Mean Final Price: ${}'.format(round(float(simulations.mean()), 2)))
    # Variance of the price (with 99% confidence interval)
    plt.figtext(x = 0.6, y = 0.6, 
		s = 'VaR(0.99): ${}'.format(round(float(start_price - q), 2)))
    # Disply 1% quantile
    plt.figtext(x = 0.15, y = 0.6,
		s = 'q(0.99): {}'.format(round(float(q), 2)))
    # Plot a line at the 1% quantile
    plt.axvline(x = q, linewidth = 4, color = 'r')
    # Title
    plt.title('Final Price Distribution for {} after {} days'.format(stock, days),
	     weight = 'bold')

    end_time = time.time()
    print 'Time taken to run {} simulations for {}: {}'.format(runs, stock, (end_time - start_time))

    plt.show()
