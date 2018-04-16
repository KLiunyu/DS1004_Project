import numpy as np
from math import sqrt
from operator import add
from pyspark.mllib.clustering import KMeans, KMeansModel
from csv import reader
from pyspark.sql import SparkSession

def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))
 
def addclustercols(x):
    point = np.array(x[4], x[5], x[6], x[7], x[8])
    center = clusters.centers[0]
    mindist = sqrt(sum([y**2 for y in (point - center)]))
    cl = 0
    for i in range(1,len(clusters.centers)):
        center = clusters.centers[i]
        distance = sqrt(sum([y**2 for y in (point - center)]))
        if distance < mindist:
            cl = i
            mindist = distance
    clcenter = clusters.centers[cl]
    return (x[0], x[1], x[2], x[3], x[5], x[6], x[7], x[8] int(cl), float(mindist))

def inclust(x, t):
    cl = x[8]
    distance = x[9]
    if float(distance) > float(t):
        cl = -1
    return (x[0], x[1], x[2], x[3],x[4], x[5], x[6], x[7], int(cl), float(distance))
    
if __name__ == '__main__':
    lines = sc.textFile(sys.argv[1], 1)
    target_data = lines.mapPartitions(lambda x: reader(x))
    
    filtered_data = target_data.filter(lambda x: np.count_nonzero(np.array(float(x[3]), float(x[5]), float(x[9]), float(x[11])]))>0)
    selected_data = filtered_data.map(lambda x: np.array(x[0], x[1], x[2], float(x[3]), x[4], float(x[5]), float(x[9]), float(x[11])]))
    df = spark.createDataFrame(selected_data, ('Year', 'Name','Pos', 'Team','Age', 'G', 'TSR', 'FTR'))
    df.createOrReplaceTempView("df")
    avf = spark.sql("SELECT COUNT(Pos)/(SELECT COUNT(*) FROM df) AS Pos_avf, Pos FROM df GROUP BY Pos")
    avf.createOrReplaceTempView("avf")
    transformed_df = spark.sql("SELECT Year, Name, Team, Pos, Pos_avf, Age,G, TSR, FTR FROM df INNER JOIN df ON avf.Pos = df.Pos")
    transformed_df.createOrReplaceTempView("transformed_df")
    cluster_df = spark.sql("SELECT Pos_avf, Age/MEAN(AGE) AS Age_scaled, G/MEAN(G) AS G_scaled, TSR/MEAN(TSR) AS TSR_scaled, FTR/MEAN(FTR) AS FTR_scaled FROM transformed_df ")
    cluster_df = cluster_df.rdd
    
    clusters = KMeans.train(cluster_df, 6, maxIterations=10, initializationMode="random")
    for i in range(0,len(clusters.centers)):
        print("cluster " + str(i) + ": " + str(clusters.centers[i]))
    
    processed_df = transformed_df.rdd.map(lambda x: addclustercols(x))
    
    outlier_detection = processed_df.map(lambda x: inclust(x,5))
    detection_df = spark.createDataFrame(outlier_detection, ('Year', 'Name','Pos', 'Team','Age', 'G', 'TSR', 'FTR', 'cluster', 'distance'))
    df.createOrReplaceTempView("")
    
    outliers = spark.sql("SELECT Year, Name, Team, Pos, Age, G, TSR, FTR FROM detection WHERE cluster=-1")
    non-outliers = spark.sql("SELECT Year, Name, Team, Pos, Age, G, TSR, FTR FROM detection WHERE cluster!=-1")
