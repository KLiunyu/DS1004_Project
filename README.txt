2018/05/13:
The null/outliers detection algorithm can be run by submit the job to spark under following format:

spark-submit --conf spark.pyspark.python=/share/apps/python/3.4.4/bin/python avf.py file_name
spark-submit --conf spark.pyspark.python=/share/apps/python/3.4.4/bin/python improved_kmean.py file_name