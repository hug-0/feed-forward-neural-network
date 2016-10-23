# Author: Hugo Nordell
# Copyright 2016
# Simple script to import csv data that represents X features and y outputs for classification problems. No headers, and y should be last column.
# usage: term$ python setLabels.py data_file.csv
import csv
import sys
import os

inName = str(sys.argv[1])
outName = inName.split('.')[0]+'_numerics.csv'

if (os.path.isabs(inName)):
	print 'Only relative paths are supported. E.g. "myfile.csv"'
	exit(0)

with open(inName, 'r') as infile, open(outName, 'wb') as outfile, open('./'+inName.split('.')[1]+'_num_labels.csv','wb') as num_labels:
	reader = csv.reader(infile, delimiter=',')
	writer = csv.writer(outfile, delimiter=',')
	l_writer = csv.writer(num_labels, delimiter=',')
	labels = []
	for row in reader:
		if (type(row[len(row)-1]) is str and row[len(row)-1] not in labels):
			labels.append(row[len(row)-1])
	l_writer.writerow([len(labels)])
	infile.seek(0) # Reset to header
	new_labels = [i for i in range(1,len(labels)+1)]
	for row in reader:
		cmp = row[len(row)-1]
		for i in range(0, len(labels)):
			if cmp == labels[i]:
				row[len(row)-1] = new_labels[i]
		writer.writerow(row)
	print 'Number of unique labels', len(labels)
