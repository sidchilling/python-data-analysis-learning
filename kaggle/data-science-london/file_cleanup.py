import csv

# The input files train.csv and trainLabels.csv does not have headers, so we
# will add it

with open('train_new.csv', 'w') as out_file:
    writer = csv.writer(out_file, delimiter = ',', 
		       quotechar = '"', quoting = csv.QUOTE_ALL)
    num_rows = 0
    with open('train.csv', 'r') as in_file:
	reader = csv.reader(in_file, delimiter = ',')
	index = 0
	for row in reader:
	    if index == 0:
		# we need to add the header column
		header_col = []
		for feature_num in range(0, len(row)):
		    header_col.append('feat_{}'.format(feature_num + 1))
		writer.writerow(header_col)
		index = index + 1
		num_rows = num_rows + 1
	    writer.writerow(row)
	    num_rows = num_rows + 1

print 'Num Rows Written: {}'.format(num_rows)

# now we will do the same thing for trainLabels.csv

with open('trainLabels_new.csv', 'w') as out_file:
    writer = csv.writer(out_file, delimiter = ',',
		       quotechar = '"', quoting = csv.QUOTE_ALL)
    num_rows = 0
    with open('trainLabels.csv', 'r') as in_file:
	reader = csv.reader(in_file, delimiter = ',')
	index = 0
	for row in reader:
	    if index == 0:
		header_col = ['target']
		writer.writerow(header_col)
		index = index + 1
		num_rows = num_rows + 1
	    writer.writerow(row)
	    num_rows = num_rows + 1

print 'Num Rows Written: {}'.format(num_rows)