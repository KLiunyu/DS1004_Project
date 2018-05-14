#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 14:20:56 2018

@author: sunan
"""
from pyspark import SparkContext
from operator import add
import pandas as pd
import numpy as np
import sys
sc = SparkContext()


def get_categ_column_list(file):
	df = pd.DataFrame.from_csv(file, sep='\t', index_col=None)
	name_lst = list(df.select_dtypes(include=['object']).columns)
	index_lst = [df.columns.get_loc(i) for i in name_lst]
	
	return index_lst, name_lst

def parseDevelopmentDatasetLine(line):
	line = line.split('\t')
	retval = []

	for i in range(len(index_lst)):
		key = name_lst[i]
		value = line[index_lst[i]]
		retval.append((key, value))
		
	return retval    

def attributeValueFrequencyScorer(line):
	"""
	Connects each line to attribute value frequency score (up to constant factor)
 
	:param  line: line turned to list of (label, value) tuples
	:returns: original line and attribute value frequency score.
	"""
 
	# Score each line and create a tuple (line, score)
	score = 0
	for item in line:
	# item[0] is attribute id
	# item[1] is value of the attribute
 		score += counts[str(item[0])][str(item[1])]
 
	return (line, score)


if __name__ == '__main__':
	index_lst, name_lst = get_categ_column_list(sys.argv[1])

 
	data = sc.textFile(sys.argv[1])
	header = data.first()
	data = data.filter(lambda row: row != header)
	parsed_data = data.map(parseDevelopmentDatasetLine)
	
 
	# Count the number of times different attribute values
	# are present in the dataset.
	
	grouped_by_attribute_value_labels = parsed_data.flatMap(lambda x: x).reduceByKey(lambda x,y: x + '\t' + y )
	keys = grouped_by_attribute_value_labels.keys().collect()
	
	# Count for all possible values for each attribute value
	counts = {}
	for key in keys:
		# Turn tuples to dict
		values = grouped_by_attribute_value_labels.filter(lambda x: x[0] == key).flatMap(lambda x: x[1].split('\t')).map(lambda x: (x, 1)).reduceByKey(add).collect()
		counts[str(key)] = dict(values)
 
	# Go through the dataset and score each item
	avf_scored_data = parsed_data.map(attributeValueFrequencyScorer).sortBy(lambda x: x[1])
	
	outliers = avf_scored_data.take(10) # Print 10 most likely outliers
	for item in outliers:
		print (item)
 
	sc.parallelize(outliers).saveAsTextFile('categorical_outliers.out')


