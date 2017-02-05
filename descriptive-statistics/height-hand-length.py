import numpy as np
import pandas as pd

import matplotlib as mtp
mtp.use('TkAgg')
import matplotlib.pyplot as plt

df = pd.read_csv('height-hand-length-data.csv')
print df.head()

male_df = df[df['Gender'] == 'M']
female_df = df[df['Gender'] == 'F']

print male_df.head()
print female_df.head()

male_x = np.ravel(male_df['Hand length'])
male_y = np.ravel(male_df['Height'])
male_m, male_b = np.polyfit(x = male_x, y = male_y, deg = 1)

female_x = np.ravel(female_df['Hand length'])
female_y = np.ravel(female_df['Height'])
female_m, female_b = np.polyfit(x = female_x, y = female_y, deg = 1)

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(x = male_x, y = male_y, c = 'b', marker = 's', label = 'Male')
ax1.scatter(x = female_x, y = female_y, c = 'r', marker = 'o', label = 'Female')
ax1.plot(male_x, male_m * male_x + male_b, c = 'b')
ax1.plot(female_x, female_m * female_x + female_b, c = 'r')

plt.legend(loc = 'upper left')
plt.show()

katie_hand_length = 6.75
print 'Katie\'s Height: {}'.format(female_m * katie_hand_length + female_b)