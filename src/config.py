# embedders
from pyspark.ml.feature import Word2Vec, HashingTF

#classifiers
from pyspark.ml.classification import RandomForestClassifier, MultilayerPerceptronClassifier

#evaluator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#define number of utterances to read in
n = 65

#define train/test split
split = [0.80, 0.20]

#define the model for converting utterances to vectors
embedder = Word2Vec(minCount=0, inputCol="utterance", outputCol="vector")

#define the model for predicting intents from embedded utterances
classifier = RandomForestClassifier(labelCol="indexedlabel", featuresCol="vector")

#define the number of folds used in cross validation
k = 5

#define the parameter grid over which we will use cross validation to find the best model
paramGrid = ParamGridBuilder().addGrid(embedder.vectorSize, [5]).addGrid(classifier.numTrees, [50]).build()

#define the metric of evaluation for the classifier
evaluator = MulticlassClassificationEvaluator(labelCol="indexedlabel", predictionCol="prediction", metricName="accuracy")
