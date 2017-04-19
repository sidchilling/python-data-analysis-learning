import numpy as np
import math

values = [33219, 36254, 38801, 46335, 46840, 47596, 55130, 56863, 78070, 88830]
values = np.array(values)

mean = np.mean(values)

print "Mean: {}".format(mean)

deviations = []
for val in values:
    deviation = val - mean
    deviations.append(deviation)
    print "Val: {}, Deviation: {}".format(val, deviation)

deviations = np.array(deviations)
mean_deviation = np.mean(deviations)
print "Mean Deviation: {}".format(mean_deviation)

abs_deviations = []
for val in deviations:
    if val < 0:
        val = (-1) * val
    abs_deviations.append(val)

abs_deviations = np.array(abs_deviations)
print "Average Absolute Deviation: {}".format(np.mean(abs_deviations))

squared_deviations = []
for val in deviations:
    sq = val * val
    squared_deviations.append(sq)
    print "{} : {}".format(val, sq)

squared_deviations = np.array(squared_deviations)
print "Variance: {}".format(np.mean(squared_deviations))
print "Std: {}".format(math.sqrt(np.mean(squared_deviations)))

social_network_values = [38946, 43420, 49191, 50430, 50557, 52580, 53595, 54135, 60181, 62076]
social_network_values = np.array(social_network_values)
print "Social Network STD: {}".format(np.std(social_network_values))

sample = [18, 20, 18, 23, 22, 17, 21, 15, 21]
sample = np.array(sample)
print "STD for sample: {}".format(np.std(sample))