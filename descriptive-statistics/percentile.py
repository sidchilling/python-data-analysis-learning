from __future__ import division

import numpy as np

def find_percentile(scores, p):
    scores = sorted(scores)

    r = (p / 100) * (len(scores) + 1)

    if r.is_integer():
	return scores[int(r) - 1]

    # r is not an integer
    return scores[int(r) - 1] + (scores[int(r)] - scores[int(r) - 1]) * (r - int(r))

scores = [3, 5, 7, 8, 9, 11, 13, 15]
print '{} : {}'.format(find_percentile(scores = scores, p = 25),
		      np.percentile(np.array(scores), q = 25))
scores = [4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 9, 9, 9, 10, 10, 10]
print '{} : {}'.format(find_percentile(scores = scores, p = 25),
		      np.percentile(np.array(scores), q = 25))
scores = [4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 9, 9, 9, 10, 10, 10]
print '{} : {}'.format(find_percentile(scores = scores, p = 85),
		      np.percentile(np.array(scores), q = 85))
scores = [2, 3, 5, 9]
print '{} : {}'.format(find_percentile(scores = scores, p = 50),
		      np.percentile(np.array(scores), q = 50))
scores = [2, 3, 5, 9, 11]
print '{} : {}'.format(find_percentile(scores = scores, p = 50),
		      np.percentile(np.array(scores), q = 50))
scores = [3, 5, 7, 9, 12, 21, 25, 30]
print '{} : {}'.format(find_percentile(scores = scores, p = 25),
		      np.percentile(np.array(scores), q = 25))
scores = [3, 5, 7, 9, 12, 21, 25, 30]
print '{} : {}'.format(find_percentile(scores = scores, p = 80),
		      np.percentile(np.array(scores), q = 80))

scores = [9.19,
	  11.76,
	   8.28,
	   9.96,
	  10.34,
	   5.40,
	  10.71,
	   8.98,
	  11.95,
	   9.85,
	  13.07,
	   7.89,
	  11.86,
	   9.52,
	   9.14,
	  13.52,
	  12.70,
	   8.04,
	   8.33,
	   8.56,
	   8.37,
	   8.08,
	  11.59,
	  10.63,
	  10.34,
	  12.57,
	  11.10,
	   7.96,
	   9.10,
	  14.10,
	  11.47,
	   5.96,
	   9.28,
	  10.49,
	  12.84,
	   9.42,
	   8.20,
	  11.68,
	  10.15,
	  12.73,
	  10.00,
	   8.36,
	   8.94]

print '{} : {}'.format(find_percentile(scores = scores, p = 75),
		      np.percentile(np.array(scores), q = 75))
print '{} : {}'.format(find_percentile(scores = scores, p = 25),
		      np.percentile(np.array(scores), q = 25))