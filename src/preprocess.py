##################################################################################
# start a local spark session
# https://spark.apache.org/docs/0.9.0/python-programming-guide.html
# spark-submit --packages com.databricks: spark - csv_2.10: 1.4.0
##################################################################################
from pyspark import SparkContext, SparkConf, SQLContext
conf = SparkConf()

#set app name
conf.set("spark.app.name", "train classifier")
#Run Spark locally with as many worker threads as logical cores on your machine (cores X threads).
conf.set("spark.master", "local[*]")
#number of cores to use for the driver process (only in cluster mode)
conf.set("spark.driver.cores", "1")
#Limit of total size of serialized results of all partitions for each Spark action (e.g. collect)
conf.set("spark.driver.maxResultSize", "1g")
#Amount of memory to use for the driver process
conf.set("spark.driver.memory", "1g")
#Amount of memory to use per executor process (e.g. 2g, 8g).
conf.set("spark.executor.memory", "2g")

#pass configuration to the spark context object along with code dependencies
sc = SparkContext(conf=conf, pyFiles=[])
from pyspark.sql.session import SparkSession
spark = SparkSession(sc)
sqlContext = SQLContext(sc)
##################################################################################

#modules for transforming input data :)
from pyspark.ml.feature import SQLTransformer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler

#use pipeline because its awesome
from pyspark.ml import Pipeline

#read csv file into pyspark df
df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('../data/test.csv')
df =df.limit(300)
print "DF \n \n \n"
print df.collect()

#varous sql transformers defined and stored in a list
sql1 = SQLTransformer(statement="SELECT depot_code, linegroup_no, from_stop_group, to_stop_group, from_act_leave_time FROM __THIS__")
sql_transformations = [sql1]

#select categorical cols
cols = ['from_stop_group', 'to_stop_group']

#index strings in categorical cols to numerical categoricals
indexers = [
    StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c))
    for c in cols
]

#one hot encode indexed categorical (this drops the last category)
encoders = [
    StringIndexer(
        inputCol=indexer.getOutputCol(),
        outputCol="{0}_encoded".format(indexer.getOutputCol()))
    for indexer in indexers
]

#assemble features into single column for ml, specify dense vector format which MXNet requires
assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders], 
                            outputCol="features")

#add everything into a pipeline
pipeline = Pipeline(stages=sql_transformations + indexers + encoders + [assembler])

#fit the pipeline to the df
pipeline = pipeline.fit(df)

#transform from input to output
rdd = pipeline.transform(df).features.rdd

frequencyDenseVectors = rdd.map(lambda vector: DenseVector(vector.toArray())

#write to resulting df to a csv in results directory
import pandas as pd
df.toPandas().to_csv('../data/mycsv.csv')
