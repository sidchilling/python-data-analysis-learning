from election_prepare_data import *

# Visualize Affiliation
sns.factorplot(x = 'Affiliation', data = poll_df, kind = 'count')
plt.show()

sns.factorplot(x = 'Affiliation', hue = 'Population', data = poll_df,
	       kind = 'count')
plt.show()

avg = DataFrame(poll_df.mean())
avg.drop(labels = ['Number of Observations', 'Question Text', 'Question Iteration', 'Other'],
	 axis = 0, inplace = True)
print avg

std = DataFrame(poll_df.std())
std.drop(labels = ['Number of Observations', 'Question Text', 'Question Iteration', 'Other'],
	 axis = 0, inplace = True)
print std

avg.plot(yerr = std, kind = 'bar', legend = False)
plt.show()

poll_avg = pd.concat([avg, std], axis = 1)
poll_avg.columns = ['Average', 'STD']
print poll_avg

poll_df.plot(x = 'End Date', y = ['Obama', 'Romney', 'Undecided'],
	     kind = 'line', linestyle = '', marker = 'o')
plt.show()

poll_df['Difference'] = (poll_df['Obama'] - poll_df['Romney']) / 100
print poll_df.head()
poll_df = poll_df.groupby(['Start Date'], as_index = False).mean()
poll_df.plot(x = 'Start Date', y = 'Difference', kind = 'line',
	    figsize = (12, 4), marker = 'o', linestyle = '-',
	    color = 'purple')
plt.show()

# Plot when the debates happened (2012-10)
row_in = 0
xlimits = []
for date in poll_df['Start Date']:
    if date[0 : 7] == '2012-10':
	xlimits.append(row_in)
    row_in = row_in + 1
print '{} : {}'.format(min(xlimits), max(xlimits))

poll_df.plot(x = 'Start Date', y = 'Difference', kind = 'line',
	    xlim = [min(xlimits), max(xlimits)], figsize = (12, 4),
	    marker = 'o', linestyle = '-', color = 'purple')
# Mark the debate dates
start_date = '2012-10-01'
debate_dates = ['2012-10-03', '2012-10-11', '2012-10-22']
for debate in debate_dates:
    num_days = (datetime.strptime(debate, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days
    label = 'Debate ({})'.format(datetime.strptime(debate, '%Y-%m-%d').strftime('%d %b'))
    plt.axvline(x = min(xlimits) + num_days, linewidth = 2,
	       color = 'grey')
    plt.text(x = min(xlimits) + num_days + 0.2, y = 0, s = label, rotation = 90)
plt.show()
