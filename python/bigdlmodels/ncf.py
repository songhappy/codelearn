from zoo.pipeline.api.keras.layers import *
from zoo.models.recommendation import UserItemFeature
from zoo.models.recommendation import NeuralCF
from zoo.common.nncontext import init_nncontext
import matplotlib
from sklearn import metrics
from operator import itemgetter
from bigdl.dataset import movielens
from bigdl.util.common import *

sc = init_nncontext("NCF Example")
movielens_data = movielens.get_id_ratings("/tmp/movielens/")
min_user_id = np.min(movielens_data[:,0])
max_user_id = np.max(movielens_data[:,0])
min_movie_id = np.min(movielens_data[:,1])
max_movie_id = np.max(movielens_data[:,1])
rating_labels= np.unique(movielens_data[:,2])

print(movielens_data.shape)
print(min_user_id, max_user_id, min_movie_id, max_movie_id, rating_labels)

def build_sample(user_id, item_id, rating):
    sample = Sample.from_ndarray(np.array([user_id, item_id]), np.array([rating]))
    return UserItemFeature(user_id, item_id, sample)
pairFeatureRdds = sc.parallelize(movielens_data)\
    .map(lambda x: build_sample(x[0], x[1], x[2]-1))
pairFeatureRdds.take(3)
trainPairFeatureRdds, valPairFeatureRdds = pairFeatureRdds.randomSplit([0.8, 0.2], seed= 1)
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
        batch_size= 8000,
        validation_data=val_rdd)

ncf.save_model("../save_model/movie_ncf.zoomodel", over_write=True)
#
weights = ncf.get_weights()
# print(weights)
print(len(weights))

for i, weight in enumerate(weights):
    print(i)
    print(weight.shape)
#
loaded = ncf.load_model("../save_model/movie_ncf.zoomodel")
user_embed = loaded.get_weights()[0]
print(user_embed.shape)
#
item_embed = loaded.get_weights()[1]
print(item_embed.shape)