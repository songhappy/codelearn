from zoo.pipeline.api.keras.layers import *
from zoo.models.recommendation import UserItemFeature
from zoo.models.recommendation import NeuralCF
from zoo.common.nncontext import init_nncontext
import matplotlib
from sklearn import metrics
from operator import itemgetter
from bigdl.dataset import movielens
from bigdl.util.common import *
import random
import pickle

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, array, broadcast, log, explode, struct, collect_list

sc = init_nncontext("NCF evaluation")
spark = SparkSession.builder \
    .master("local[1]") \
    .appName("SparkByExamples.com") \
    .getOrCreate()

movielens_data = movielens.get_id_ratings("./data/movielens/")

rate_file = "./data/movielens/ml-1m/ratings.dat"
ratings = []
with open(rate_file) as infile:
    for cnt, line in enumerate(infile):
        x = line.strip().split("::")
        y = list(map(lambda item: int(item), x))
        # y[3] = datetime.datetime.fromtimestamp(y[3])
        ratings.append(y)

min_user_id = np.min(movielens_data[:,0])
max_user_id = np.max(movielens_data[:,0])
min_movie_id = np.min(movielens_data[:,1])
max_movie_id = np.max(movielens_data[:,1])
rating_labels= np.unique(movielens_data[:,2])

# open a file, where you ant to store the data
with open("./data/movielens/train.pickle", "rb") as f:
    train_data = pickle.load(f)
with open("./data/movielens/test.pickle", "rb") as f:
    test_data = pickle.load(f)

print(min_user_id, max_user_id, min_movie_id, max_movie_id, rating_labels)

def build_sample(user_id, item_id, rating):
    sample = Sample.from_ndarray(np.array([user_id, item_id]), np.array([rating]))
    return UserItemFeature(user_id, item_id, sample)

trainPairFeatureRdds = sc.parallelize(train_data)\
    .map(lambda x: build_sample(x[0], x[1], x[2]))
valPairFeatureRdds = sc.parallelize(test_data) \
    .map(lambda x: build_sample(x[0], x[1], x[2]))
valPairFeatureRdds.cache()

train_rdd= trainPairFeatureRdds.map(lambda pair_feature: pair_feature.sample)
val_rdd= valPairFeatureRdds.map(lambda pair_feature: pair_feature.sample)
val_rdd.persist()

ncf = NeuralCF(user_count=max_user_id,
               item_count=max_movie_id,
               class_num=5,
               hidden_layers=[20, 10],
               include_mf = False)

ncf.compile(optimizer= "adam",
            loss= "sparse_categorical_crossentropy",
            metrics=['accuracy'])

loaded = ncf.load_model("./save_model/movie_ncf1.zoomodel") #old

predictions = ncf.predict_classes(val_rdd).collect()
#print(predictions[1:10000])

recommendations = ncf.recommend_for_user(valPairFeatureRdds, 10)
#for rec in recommendations.take(5): print(rec)
user_items_rec10 = recommendations.map(lambda x: [x.user_id, x.item_id]).collect()
import pandas

rec_df = pandas.DataFrame(user_items_rec10, columns = ['uid', 'mid'])
rec_df = rec_df.groupby('uid', as_index=False).agg(lambda x: list(x))
print("length:")
print(len(rec_df))

test_df = pandas.DataFrame(test_data, columns=['uid', 'mid', 'rate', 'timestamp'])
test_df['rate'] = test_df['rate'] + 1
test_df['Rank'] = test_df.groupby('uid', as_index=False)['rate'].transform(lambda x: x.rank(ascending=False, method="first"))
test_df = test_df[test_df['Rank'] < 11]
#print(test_df.head(100))
test_df = test_df[['uid', 'mid']]
test_df = test_df.groupby('uid', as_index=False).agg(lambda x: list(x))

print("length:")
print(len(test_df))

rec_df = spark.createDataFrame(rec_df)
rec_df.show(10)

test_df = spark.createDataFrame(test_df)
test_df.show(10)

joined = rec_df.withColumnRenamed('mid', 'midrec').join(test_df, on=['uid'])
joined.show(10)

def precision(prediction, groundtruth):
    sum = 0.0
    for ele in prediction:
        if ele in groundtruth:
            sum = sum + 1
    return sum/(len(prediction))

precision_udf = udf(lambda c1, c2: precision(c1, c2))

joined=joined.withColumn("precision", precision_udf('midrec', 'mid'))
#def precision():
joined.show(10, False)
from pyspark.sql.functions import mean as _mean

stats = joined.select(_mean(col('precision')).alias('mean')).collect()
mean = stats[0]['mean']
print("precision @ k:", mean)
