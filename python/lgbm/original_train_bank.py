# coding: utf-8

import lightgbm as lgb
from bigdl.orca import init_orca_context, OrcaContext
from sklearn.metrics import accuracy_score, roc_auc_score

print('Loading data...')
# load or create your dataset)
input_path = "/home/arda/intelWork/data/bankruptcy/data.csv"
sc = init_orca_context("local", cores=2)
spark = OrcaContext.get_spark_session()
df = (spark.read.format("csv")
      .option("header", True)
      .option("inferSchema", True)
      .load(input_path))

train, test = df.randomSplit([0.20, 0.80], seed=1)
train = train.toPandas()
test = test.toPandas()
print(train.head())
y_train = train["Bankrupt?"]
y_test = test["Bankrupt?"]
X_train = train.drop(["Bankrupt?"], axis=1)
X_test = test.drop(["Bankrupt?"], axis=1)

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
