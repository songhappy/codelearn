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
import argparse
from bigdl.dllib.nnframes.tree_model import LightGBMClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

input_path = "/home/arda/intelWork/data/bankruptcy/data.csv"
sc = init_orca_context("local")
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
df.show(10)
train, test = df.randomSplit([0.20, 0.80], seed=1)
from pyspark.ml.feature import VectorAssembler

feature_cols = df.columns[1:]
featurizer = VectorAssembler(inputCols=feature_cols, outputCol="features")
train = featurizer.transform(train)["Bankrupt?", "features"]
test = featurizer.transform(test)["Bankrupt?", "features"]

preprocess = time.time()
params = {"boosting_type": "gbdt", "num_leaves": 70, "learning_rate": 0.3,
          "min_data_in_leaf": 20, "objective": "binary",
          'num_iterations': 10000,
          'max_depth': 14,
          'lambda_l1': 0.01,
          'lambda_l2': 0.01,
          'bagging_freq': 5,
          'max_bin': 255,
          'early_stopping_round': 20
          }

params = {"objective": "binary", 'num_iterations': 30}

for learning_rate in [0.1]:
    for max_depth in [14]:
        for num_iterations in [10000]:
            # params.update({"learning_rate": learning_rate, "max_depth": max_depth, "num_iterations": num_iterations})

            model = LightGBMClassifier(params)
            # model = LightGBMClassifier()
            model.setLabelCol("Bankrupt?")

            model = model.fit(train)
            predictions = model.transform(test)
            predictions.cache()
            predictions.show(5, False)
            evaluator = BinaryClassificationEvaluator(labelCol="Bankrupt?",
                                                      rawPredictionCol="rawPrediction")
            auc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})

            evaluator2 = MulticlassClassificationEvaluator(labelCol="Bankrupt?",
                                                           predictionCol="prediction")
            acc = evaluator2.evaluate(predictions, {evaluator2.metricName: "accuracy"})
            print(params)
            print("AUC: %.2f" % (auc * 100.0))
            print("Accuracy: %.2f" % (acc * 100.0))

            predictions.unpersist(blocking=True)

end = time.time()
print("training time: %.2f" % (end - preprocess))
stop_orca_context()
