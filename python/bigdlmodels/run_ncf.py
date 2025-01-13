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

sc = init_nncontext("NCF Example")
movielens_data = movielens.get_id_ratings("./data/movielens/")

rate_file = "./data/movielens/ml-1m/ratings.dat"
ratings = []
with open(rate_file) as infile:
    for cnt, line in enumerate(infile):
        x = line.strip().split("::")
        y = list(map(lambda item: int(item), x))
        y = [y[0] - 1, y[1] - 1, y[2] - 1, y[3]]
        # y[3] = datetime.datetime.fromtimestamp(y[3])
        ratings.append(y)

min_user_id = np.min(movielens_data[:,0])
max_user_id = np.max(movielens_data[:,0])
min_movie_id = np.min(movielens_data[:,1])
max_movie_id = np.max(movielens_data[:,1])
rating_labels= np.unique(movielens_data[:,2])

random.shuffle(ratings)
train_data = ratings[:800000]
test_data = ratings[800000:]
# open a file, where you ant to store the data
with open("./data/movielens/train.pickle", "wb") as f:
    pickle.dump(train_data, f)
with open("./data/movielens/test.pickle", "wb") as f:
    pickle.dump(test_data, f)

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

ncf.fit(train_rdd,
        nb_epoch= 10,
        batch_size= 8800,
        validation_data=val_rdd)

ncf.save_model("./save_model/movie_ncf2.zoomodel", over_write=True)  #new
#
weights = ncf.get_weights()
# print(weights)
print(len(weights))

for i, weight in enumerate(weights):
    print(i)
    print(weight.shape)
#
loaded = ncf.load_model("./save_model/movie_ncf2.zoomodel") #old
user_embed = loaded.get_weights()[0]
print(user_embed.shape)
#
item_embed = loaded.get_weights()[1]
print(item_embed.shape)
