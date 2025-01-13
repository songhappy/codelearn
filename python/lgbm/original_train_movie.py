# coding: utf-8

import lightgbm as lgb
import pandas as pd
from bigdl.dllib.feature.dataset import movielens
from bigdl.friesian.feature import FeatureTable
from bigdl.orca import init_orca_context, OrcaContext
from pyspark.ml.linalg import DenseVector, VectorUDT
from sklearn.metrics import accuracy_score, roc_auc_score

print('Loading data...')
# load or create your dataset)
data_dir = "/home/arda/intelWork/data/movielens"
sc = init_orca_context("local", cores=2)
spark = OrcaContext.get_spark_session()

num_cols = ["user_visits", "user_mean_rate", "item_visits", "item_mean_rate"]
cat_cols = ["gender", "age", "occupation", "zip"]
features = num_cols + cat_cols

ratings = movielens.get_id_ratings(data_dir)
ratings = pd.DataFrame(ratings, columns=["user", "item", "rate"])
ratings_tbl = FeatureTable.from_pandas(ratings).cast(["user", "item", "rate"], "int")
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
    .apply("features", "features", lambda x: DenseVector(x), VectorUDT())
train.show(5, False)

test = test_tbl.merge_cols(features, "features") \
    .select(["label", "features"]) \
    .apply("features", "features", lambda x: DenseVector(x), VectorUDT())
test.show(5, False)

df_train, df_test = train.to_pandas(), test.to_pandas()
X_train = pd.DataFrame(df_train["features"].to_list())
X_test = pd.DataFrame(df_test["features"].to_list())
y_train = df_train["label"]
y_test = df_test["label"]

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

params = {"objective": "binary", 'num_iterations': 100}

print('Starting training...')
# train
gbm = lgb.train(params,
                lgb_train,
                valid_sets=lgb_eval)

print('Saving model...')
# save model to file
gbm.save_model('model.txt')

print('Starting predicting...')
# predict
y_pred = gbm.predict(X_test)
auc = roc_auc_score(y_test, y_pred)

# rounding the values
y_pred = y_pred.round(0)
# converting from float to integer
y_pred = y_pred.astype(int)

print(y_pred)
acc = accuracy_score(y_test, y_pred)
# roc_auc_score metric
# eval
print("AUC: %.2f" % (auc * 100.0))
print("Accuracy: %.2f" % (acc * 100.0))
