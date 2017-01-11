import numpy as np
import pandas as pd
from pandas import DataFrame, Series

import matplotlib as mtp
mtp.use('TkAgg')

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

dataset = np.random.randn(25)

x_min = dataset.min() - 2
x_max = dataset.max() - 2
x_axis = np.linspace(start = x_min, stop = x_max, num = 100)

bandwidth = ((4 * dataset.std() ** 5) / (3 * len(dataset))) ** 0.2

kernel_list = []
for data_point in dataset:
    # Create a kernel for each point and append it to `kernel_list`
    kernel = stats.norm(data_point, bandwidth).pdf(x_axis)
    kernel_list.append(kernel)

    # scale for plotting
    kernel = kernel / kernel.max()
    kernel = kernel * 0.4

    #plt.plot(x_axis, kernel, color = 'grey', alpha = 0.5)

sum_of_kde = np.sum(kernel_list, axis = 0)
fig = plt.plot(x_axis, sum_of_kde, color = 'indianred', clip_on = False,
	      fillstyle = 'full')

sns.rugplot(dataset)
plt.yticks([])
plt.suptitle('Sum of the basis functions')
# plt.ylim(0, 1)
plt.show()

# Doing all of the above using seaborn
sns.kdeplot(dataset)
plt.show()

sns.rugplot(dataset, color = 'black')
for bw in np.arange(0.5, 2, 0.25):
    sns.kdeplot(dataset, bw = bw, lw = 1.8, label = bw)
plt.show()

# Different kernel types of graphs
kernel_options = ['biw', 'cos', 'epa', 'gau', 'tri', 'triw']

for kern in kernel_options:
    sns.kdeplot(dataset, kernel = kern, label = kern, shade = True)
plt.show()

sns.kdeplot(dataset, vertical = True) # plotting on the vertical axis
plt.show()

sns.kdeplot(dataset, cumulative = True)
plt.show()

mean = [0, 0]
cov = [[1, 0], [0, 100]]
dataset2 = np.random.multivariate_normal(mean, cov, 1000)
df = DataFrame(dataset2, columns = ['X', 'Y'])
sns.kdeplot(df)
plt.show()

sns.kdeplot(df.X, df.Y, shade = True)
plt.show()

# applying bandwidths
sns.kdeplot(df, bw = 1)
plt.show()

sns.jointplot('X', 'Y', df, kind = 'kde')
plt.show()