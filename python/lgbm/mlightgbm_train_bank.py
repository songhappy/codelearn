#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from bigdl.orca import init_orca_context, stop_orca_context, OrcaContext
import time
import os

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Please use 0.10.0 version for Spark3.2 and 0.9.5-13-d1b51517-SNAPSHOT version for Spark3.1
spark_conf = {"spark.app.name": "recsys-lightGBM",
              "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
              "spark.driver.memoryOverhead": "8G",
              "spark.jars.packages":
                  "com.microsoft.azure:synapseml_2.12:0.9.5-13-d1b51517-SNAPSHOT",
              "spark.jars.repositories": "https://mmlspark.azureedge.net/maven"}

input_path = "/home/arda/intelWork/data/bankruptcy/data.csv"
sc = init_orca_context("local", conf=spark_conf, cores=32)
spark = OrcaContext.get_spark_session()
df = (
    spark.read.format("csv")
        .option("header", True)
        .option("inferSchema", True)
        .load(input_path)
)
print("records read: " + str(df.count()))
print("Schema: ")
df.printSchema()
train, test = df.randomSplit([0.50, 0.50], seed=1)

from pyspark.ml.feature import VectorAssembler

feature_cols = df.columns[1:]
featurizer = VectorAssembler(inputCols=feature_cols, outputCol="features")
train = featurizer.transform(train)["Bankrupt?", "features"]
test = featurizer.transform(test)["Bankrupt?", "features"]

from synapse.ml.lightgbm import LightGBMClassifier

estimator = LightGBMClassifier(objective="binary", featuresCol="features", labelCol="Bankrupt?")

estimator.setNumIterations(100)
estimator.setObjective("binary")

model = estimator.fit(train)
predictions = model.transform(test)
predictions.show(5)
print(predictions.count())

evaluator = BinaryClassificationEvaluator(labelCol="Bankrupt?",
                                          rawPredictionCol="rawPrediction")
auc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})

evaluator2 = MulticlassClassificationEvaluator(labelCol="Bankrupt?",
                                               predictionCol="prediction")
acc = evaluator2.evaluate(predictions, {evaluator2.metricName: "accuracy"})

print("AUC: %.2f" % (auc * 100.0))
print("Accuracy: %.2f" % (acc * 100.0))

stop_orca_context()
