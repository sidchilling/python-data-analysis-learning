import numpy as np
from termcolor import colored, cprint

values = [38946, 43420, 49191, 50430, 50557, 52580, 53595, 54135, 60181, 10000000]
values = np.array(values)
values = np.sort(values)

cprint("---- Manual Method ---", "yellow")
midIndex = len(values) / 2.0
firstHalf = values[0 : int(midIndex)]
secondHalf = values[int(midIndex) : len(values)]

q1 = firstHalf[int(len(firstHalf) / 2.0)]
q3 = secondHalf[int(len(secondHalf) / 2.0)]
res = "{}{}{}{}".format(colored("[Q1] ", "red"), q1, colored(" [Q3] ", "red"), q3)
print res

iqr = q3 - q1
print "{} {}".format(colored("[IQR]", "red"), iqr)

print colored("In-built Numpy Functions", "yellow")

q1 = np.percentile(values, [25])
q3 = np.percentile(values, [75])
print "{}{}{}{}".format(colored("[Q1] ", "red"), q1, colored(" [Q3] ", "red"), q3)
iqr = q3 - q1
print "{} {}".format(colored("[IQR]", "red"), iqr)