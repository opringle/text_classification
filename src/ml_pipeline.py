##################################################################################
# configure a local spark session
# https://spark.apache.org/docs/0.9.0/python-programming-guide.html
##################################################################################
from pyspark import SparkContext, SparkConf
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
##################################################################################

#read data from the hdfs folder into a pyspark RDD
path = "hdfs://localhost:9000/user/opringle/transit_data.txt"
rdd = spark.read.json(path)

#split to train and test sets
test, train = rdd.randomSplit(weights=[0.3, 0.7], seed=1)

#design a pipeline for conducting ML on the data
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler

#configure an ML pipeline
ignore = []
assembler = VectorAssembler(inputCols=[x for x in rdd.columns if x not in ignore],outputCol='features')
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexed_features", maxCategories=4)
rf = RandomForestRegressor(featuresCol="indexed_features", labelCol="act_run_time")

pipeline = Pipeline(stages=[assembler, featureIndexer, rf])

# Fit the pipeline to training documents.
model = pipeline.fit(train)

# Make predictions on test documents and print columns of interest.
train_pred = model.transform(train)
test_pred = model.transform(test)

#get statistics on the test error
evaluator = RegressionEvaluator(labelCol="act_run_time", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(test_pred)
print("\n \n \n Root Mean Squared Error (RMSE) on test data = %g" % rmse)
print "\n \n \n"

#stop spark session
sc.stop()
