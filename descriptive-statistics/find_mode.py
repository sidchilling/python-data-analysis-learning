# This finds the mode of a given list of n

import numpy as np
from scipy import stats

detroit = [73, 75, 70, 72, 71, 72, 72, 72, 75, 71, 74, 71, 74, 69, 71, 73, 76, 76,
           75, 72, 75, 74, 74, 70, 75, 72, 72, 74]
san_jose = [73, 74, 73, 73, 74, 71, 77, 74, 74, 77, 71, 73, 76, 72, 71, 74, 77, 74, 75,
            75, 74, 73, 73, 74]

print "Detroit Num: {}".format(len(detroit))
print "San Jose Num: {}".format(len(san_jose))

detroit_np = np.array(detroit)
san_jose_np = np.array(san_jose)

print "{}".format(detroit_np)
print "{}".format(san_jose_np)

print "Detroit Mpde: {}".format(stats.mode(detroit_np)[0][0])
print "San Jose Mode: {}".format(stats.mode(san_jose_np)[0][0])

print "Detroit Median: {}".format(np.median(detroit_np))
print "San Jose Median: {}".format(np.median(san_jose_np))

deal_data = [0.01, 1, 5, 10, 25, 50, 75, 100, 200, 300, 400, 500, 750, 1000, 5000, 10000,
             25000, 50000, 75000, 100000, 200000, 300000, 400000, 500000, 750000, 1000000]
deal_np = np.array(deal_data)
print "Deal Mode: {}".format(stats.mode(deal_np))
median = np.median(deal_np)
print "Deal Data Median: {}".format(median)

less_than_median = []
more_than_median = []

for d in deal_data:
    if d < median:
        less_than_median.append(d)
    elif d > median:
        more_than_median.append(d)

print "Less than Median: {}".format(less_than_median)
print "More than Median: {}".format(more_than_median)
print "Num Less than Median: {}".format(len(less_than_median))
print "Num more than Median: {}".format(len(more_than_median))
deal_data_mean = np.mean(deal_np)
print "Deal Data Mean: {}".format(deal_data_mean)

num_more_than_mean = 0
for d in deal_data:
    if d > deal_data_mean:
        num_more_than_mean += 1
proportion = num_more_than_mean / float(len(deal_data))
print "Proprtion above the mean: {}".format(proportion)





