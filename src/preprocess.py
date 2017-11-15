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
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler

from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import StringType
from pyspark.sql.types import ArrayType
from pyspark.sql.types import IntegerType
from pyspark.sql.types import FloatType
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from pyspark.ml.linalg import DenseVector

#use pipeline because its awesome
from pyspark.ml import Pipeline

#select categorical/continuous cols and label
bin_cols = ['depot_code']
cat_cols = ['linegroup_no', 'from_stop_group', 'to_stop_group']
cont_cols = ["from_act_leave_time"]
label_col = []
feature_cols = bin_cols + cat_cols + cont_cols
output_feature_cols = [s + "_indexed" for s in bin_cols] + [s + "_indexed_encoded_densified" for s in cat_cols] + [s + "_scaled" for s in cont_cols]

#read csv into pyspark df, automatically infering schema
df = spark.read.csv("../data/test.csv", encoding="utf-8", inferSchema = True, header=True)# schema=df_schema)
df = df.limit(300)
print df.show(n=2)

#convert string col to double type
# from pyspark.sql.types import DoubleType
# df = df.withColumn("from_act_leave_time",df["from_act_leave_time"].cast(DoubleType()))

#varous sql transformers defined and stored in a list
sql1 = SQLTransformer(
    statement="SELECT " + ",".join(feature_cols) + " FROM __THIS__")
sql_transformations = [sql1]

#index strings in categorical and binary cols to numerical categoricals
binary_indexers = [
    StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c))
    for c in bin_cols
]

categorical_indexers = [
    StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c))
    for c in cat_cols
]

#one hot encode indexed categorical
encoders = [
    OneHotEncoder(dropLast=False,
                  inputCol=indexer.getOutputCol(),
                  outputCol="{0}_encoded".format(indexer.getOutputCol()))
    for indexer in categorical_indexers
]

#use pyspark minmaxscaler as a hack to densify a sparse vector
densifiers = [
    MinMaxScaler(min=0.0, max=1.0, 
                 inputCol=encoder.getOutputCol(),
                 outputCol="{0}_densified".format(encoder.getOutputCol()))
    for encoder in encoders
]

#scale continous columns to be between 0 and 1 for deep learning
scalers = [
    MinMaxScaler(min=0.0, max=1.0, inputCol=c,
                 outputCol="{0}_scaled".format(c))
    for c in cont_cols
]

#use SQL to select columns to keep in pipeline
col_selector = SQLTransformer(
    statement="SELECT " + ",".join(output_feature_cols) + " FROM __THIS__")


#add everything into a pipeline
pipeline = Pipeline(stages=sql_transformations + binary_indexers + categorical_indexers + encoders + densifiers + scalers + [col_selector])

#fit the pipeline to the df
pipeline = pipeline.fit(df)

#use it to transform input data 
df = pipeline.transform(df)

print df.show(n=2)

types = [f.dataType for f in df.schema.fields]

print types

#combine all feature columns into a single vector


#write to resulting df to a csv in results directory
import pandas as pd
df.toPandas().to_csv('../data/mycsv.csv')



