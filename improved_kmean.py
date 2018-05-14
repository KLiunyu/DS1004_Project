#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 14:20:56 2018
@author: Kevin`
"""

from pyspark import SparkContext
from operator import add
import pandas as pd
import numpy as np
import math
import csv
from csv import reader
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.sql import SparkSession

#from pyspark.sql import SparkSession
sc = SparkContext()
spark = SparkSession(sc)

def get_numerical_column_list(file):
    df = pd.DataFrame.from_csv(file, sep='\t', index_col=None)
    name_lst = list(df.select_dtypes(include=['float64']).columns)
    index_lst = [df.columns.get_loc(i) for i in name_lst]
    res_name_lst = list(df.select_dtypes(exclude=['float64']).columns)
    res_index_lst = [df.columns.get_loc(i) for i in name_lst]
    
    return index_lst, name_lst, res_index_lst, res_name_lst

def cvt_float(string, cnt):
    try:
        res = float(string)
    except ValueError:
        res = 0
        cnt = cnt + 1
    return(res, cnt)

def ConvertLine(line):
    line_num = line[1]
    line_val = line[0]
    cnt = 0
    for i in range(len(index_lst)):
        key = name_lst[i]
        value, cnt = cvt_float(line_val[index_lst[i]], cnt)
        line_val[index_lst[i]] = value
    retval = line_val + [line_num, cnt]
    return retval

def GetNumericalPair(line):
    retval = []
    for i in range(len(index_lst)):
        key = name_lst[i]
        value = line[index_lst[i]]
        retval.append((key, value))
    return retval

def GetValue(line):
    retval = []
    for i in range(len(line)):
        value = line[i][1]
        mean_scale = scale_dic[line[i][0]]
        retval.append(float(value)/float(mean_scale))
    return retval

def error(point):
    center = clusters.centers[clusters.predict(point)]
    return math.sqrt(sum([x**2 for x in (point - center)]))

def addclustercols(x):
    point = []
    for i in range(len(index_lst)):
        point.append(float(x[index_lst[i]])/ float(scale_dic[name_lst[i]]))
    point = np.array(point)
    center = clusters.centers[0]
    mindist = math.sqrt(sum([y**2 for y in (point - center)]))
    cl = 0
    for i in range(1,len(clusters.centers)):
        center = clusters.centers[i]
        distance = math.sqrt(sum([y**2 for y in (point - center)]))
        if distance < mindist:
            cl = i
            mindist = distance
    clcenter = clusters.centers[cl]
    return (x+ [int(cl), float(mindist)] )

def inclust(x, t):
    cl = x[-2]
    distance = x[-1]
    if float(distance) > float(t):
        cl = -1
    return (x[0:-2] + [int(cl), float(distance)])

if __name__ == '__main__':
    index_lst, name_lst, res_index_lst, res_name_lst = get_numerical_column_list('5fn4-dr26.tsv')


    raw_data = sc.textFile('5fn4-dr26.tsv', 1)
    raw_data = raw_data.mapPartitions(lambda x: reader(x, delimiter = '\t'))
    header = raw_data.first()
    data = raw_data.filter(lambda row: row != header).zipWithIndex()
    
    #Count number of Null value in each line and filter out null lines
    data = data.map(ConvertLine)
    null_line = data.filter(lambda x: x[-1] > math.floor(len(index_lst)/2)).map(lambda x: x[-2])
    null_line_num = null_line.collect()
    print("The following lines has null value:")
    for i in null_line_num:
        print(i)
    
    clean_data = data.filter(lambda x: x[-2] not in null_line_num).map(lambda x: x[0: -2])
    d_num = clean_data.count()
    
    #normalize non_null_data for better clustering performance
    numerical_pair = clean_data.map(GetNumericalPair)
    scale = numerical_pair.flatMap(lambda x: [i for i in x]).reduceByKey(add)
    scale_dic = {}
    for i in scale.collect():
        scale_dic[i[0]] = float(i[1])/float(d_num)
    
    numerical_data = numerical_pair.map(GetValue)
    
    
    # Cluster the data using K-means algorithm, find cluster center
    clusters = KMeans.train(numerical_data, 5, maxIterations=10, initializationMode="random")
    for i in range(0,len(clusters.centers)):
        print("cluster " + str(i) + ": " + str(clusters.centers[i]))
        
    #classify data to each cluster center 
    processed_data = clean_data.map(lambda x: addclustercols(x))
    
    #find outlier bound
    distance = processed_data.map(lambda x: x[-1]).sortBy(lambda x: x, ascending = False)
    #out_num = math.floor(d_num*0.001)
    out_bound = distance.take(10)[-1]
    
    
    #filter out outliers
    processed_data = processed_data.map(lambda x: inclust(x,out_bound))
    outliers = processed_data.filter(lambda x: x[-2] == -1).map(lambda x: x[0: -2])
    #outliers.take(1)

    df = spark.createDataFrame(sc.parallelize(outliers.collect()), schema = header)
    df.show()    
    
    res_lst = outliers.collect()
    myfile = open('Improved_kmean_out.csv','w')
    wr = csv.writer(myfile, quoting = csv.QUOTE_MINIMAL, quotechar = "'")
    wr.writerows(res_lst)
    myfile.close()

