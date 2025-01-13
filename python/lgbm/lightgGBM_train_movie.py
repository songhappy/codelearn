import pandas as pd
from bigdl.dllib.feature.dataset import movielens
from bigdl.dllib.nnframes.tree_model import LightGBMClassifier
from bigdl.friesian.feature import FeatureTable
from bigdl.orca import init_orca_context, stop_orca_context, OrcaContext
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.linalg import DenseVector, VectorUDT

sc = init_orca_context("local", cores=8, memory="16g")
spark = OrcaContext.get_spark_session()

# movielens data
data_dir = "/home/arda/intelWork/data/movielens"
num_cols = ["user_visits", "user_mean_rate", "item_visits", "item_mean_rate"]
cat_cols = ["gender", "age", "occupation", "zip"]
features = num_cols + cat_cols

ratings = movielens.get_id_ratings(data_dir)
ratings = pd.DataFrame(ratings, columns=["user", "item", "rate"])
ratings_tbl = FeatureTable.from_pandas(ratings) \
    .cast(["user", "item", "rate"], "int")
ratings_tbl.cache()

user_tbl = FeatureTable.read_csv(data_dir + "/ml-1m/users.dat", delimiter=":") \
    .select("_c0", "_c2", "_c4", "_c6", "_c8") \
    .rename({"_c0": "user", "_c2": "gender", "_c4": "age", "_c6": "occupation", "_c8": "zip"}) \
    .cast(["user"], "int")
user_tbl.cache()

user_tbl = user_tbl.fillna("0", "zip")

user_stats = ratings_tbl.group_by("user", agg={"item": "count", "rate": "mean"}) \
    .rename({"count(item)": "user_visits", "avg(rate)": "user_mean_rate"})
user_stats, user_min_max = user_stats.min_max_scale(["user_visits", "user_mean_rate"])

item_stats = ratings_tbl.group_by("item", agg={"user": "count", "rate": "mean"}) \
    .rename({"count(user)": "item_visits", "avg(rate)": "item_mean_rate"})
item_stats, item_min_max = item_stats.min_max_scale(["item_visits", "item_mean_rate"])

user_tbl, inx_list = user_tbl.category_encode(["gender", "age", "zip", "occupation"])

item_size = item_stats.select("item").distinct().size()
ratings_tbl = ratings_tbl.add_negative_samples(item_size=item_size, item_col="item",
                                               label_col="label", neg_num=1)

user_tbl = user_tbl.join(user_stats, on="user")
full = ratings_tbl.join(user_tbl, on="user").join(item_stats, on="item")
full, target_codes = full.target_encode(cat_cols=cat_cols, target_cols=["label"])
stats = full.get_stats(cat_cols, "max")
train_tbl, test_tbl = full.select("label", *features).random_split([0.8, 0.2])

train = train_tbl.merge_cols(features, "features") \
    .select(["label", "features"]) \
    .apply("features", "features", lambda x: DenseVector(x), VectorUDT()).repartition(10)
train.show(5, False)

test = test_tbl.merge_cols(features, "features") \
    .select(["label", "features"]) \
    .apply("features", "features", lambda x: DenseVector(x), VectorUDT()).repartition(10)
test.show(5, False)

params = {"objective": "binary", 'num_iterations': 200}

for learning_rate in [0.1]:
    for max_depth in [14]:
        for num_iterations in [10000]:
            # params.update({"learning_rate": learning_rate, "max_depth": max_depth, "num_iterations": num_iterations})

            model = LightGBMClassifier(params)
            # model = LightGBMClassifier()

            model = model.fit(train.df_scaled)
            predictions = model.transform(test.df_scaled)
            predictions.cache()
            predictions.show(5, False)
            evaluator = BinaryClassificationEvaluator(labelCol="label",
                                                      rawPredictionCol="rawPrediction")
            auc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})

            evaluator2 = MulticlassClassificationEvaluator(labelCol="label",
                                                           predictionCol="prediction")
            acc = evaluator2.evaluate(predictions, {evaluator2.metricName: "accuracy"})
            print(params)
            print("AUC: %.2f" % (auc * 100.0))
            print("Accuracy: %.2f" % (acc * 100.0))

            predictions.unpersist(blocking=True)

stop_orca_context()
