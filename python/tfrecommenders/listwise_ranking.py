# # Listwise ranking
# In [the basic ranking tutorial](basic_ranking), we trained a model that can predict ratings for user/movie pairs. The model was trained to minimize the mean squared error of predicted ratings.
# However, optimizing the model's predictions on individual movies is not necessarily the best method for training ranking models. We do not need ranking models to predict scores with great accuracy. Instead, we care more about the ability of the model to generate an ordered list of items that matches the user's preference ordering.
# Instead of optimizing the model's predictions on individual query/item pairs, we can optimize the model's ranking of a list as a whole. This method is called _listwise ranking_.
# In this tutorial, we will use TensorFlow Recommenders to build listwise ranking models. To do so, we will make use of ranking losses and metrics provided by [TensorFlow Ranking](https://github.com/tensorflow/ranking), a TensorFlow package that focuses on [learning to rank](https://www.microsoft.com/en-us/research/publication/learning-to-rank-from-pairwise-approach-to-listwise-approach/).

import pprint
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_ranking as tfr
import tensorflow_recommenders as tfrs
batch_size = 1024
# We will continue to use the MovieLens 100K dataset. As before, we load the datasets and keep only the user id, movie title, and user rating features for this tutorial. We also do some houskeeping to prepare our vocabularies.

ratings = tfds.load("movielens/100k-ratings", split="train")
movies = tfds.load("movielens/100k-movies", split="train")

ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
    "user_rating": x["user_rating"],
})
movies = movies.map(lambda x: x["movie_title"])

unique_movie_titles = np.unique(np.concatenate(list(movies.batch(1000))))
unique_user_ids = np.unique(np.concatenate(list(ratings.batch(1_000).map(
    lambda x: x["user_id"]))))


# ## Data preprocessing
# 
# However, we cannot use the MovieLens dataset for list optimization directly. To perform listwise optimization, we need to have access to a list of movies each user has rated, but each example in the MovieLens 100K dataset contains only the rating of a single movie.
# To get around this we transform the dataset so that each example contains a user id and a list of movies rated by that user. Some movies in the list will be ranked higher than others; the goal of our model will be to make predictions that match this ordering.
# To do this, we use the `tfrs.examples.movielens.movielens_to_listwise` helper function. It takes the MovieLens 100K dataset and generates a dataset containing list examples as discussed above. The implementation details can be found in the [source code](https://github.com/tensorflow/recommenders/blob/main/tensorflow_recommenders/examples/movielens.py).

tf.random.set_seed(42)

# Split between train and tests sets, as before.
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

# We sample 50 lists for each user for the training data. For each list we
# sample 5 movies from the movies the user rated.
train = tfrs.examples.movielens.sample_listwise(
    train,
    num_list_per_user=50,
    num_examples_per_list=5,
    seed=42
)
test = tfrs.examples.movielens.sample_listwise(
    test,
    num_list_per_user=1,
    num_examples_per_list=5,
    seed=42
)

# We can inspect an example from the training data. The example includes a user id, a list of 10 movie ids, and their ratings by the user.

for example in train.take(1):
  pprint.pprint(example)

class RankingModel(tfrs.Model):

  def __init__(self, loss):
    super().__init__()
    embedding_dimension = 32

    # Compute embeddings for users.
    self.user_embeddings = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_user_ids),
      tf.keras.layers.Embedding(len(unique_user_ids) + 2, embedding_dimension)
    ])

    # Compute embeddings for movies.
    self.movie_embeddings = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_movie_titles),
      tf.keras.layers.Embedding(len(unique_movie_titles) + 2, embedding_dimension)
    ])

    # Compute predictions.
    self.score_model = tf.keras.Sequential([
      # Learn multiple dense layers.
      tf.keras.layers.Dense(256, activation="relu"),
      tf.keras.layers.Dense(64, activation="relu"),
      # Make rating predictions in the final layer.
      tf.keras.layers.Dense(1)
    ])

    self.task = tfrs.tasks.Ranking(
      loss=loss,
      metrics=[
        tfr.keras.metrics.NDCGMetric(name="ndcg_metric"),
        tf.keras.metrics.RootMeanSquaredError()
      ]
    )

  def call(self, features):
    # We first convert the id features into embeddings.
    # User embeddings are a [batch_size, embedding_dim] tensor.
    user_embeddings = self.user_embeddings(features["user_id"])

    # Movie embeddings are a [batch_size, num_movies_in_list, embedding_dim]
    # tensor.
    movie_embeddings = self.movie_embeddings(features["movie_title"])
    
    # We want to concatenate user embeddings with movie emebeddings to pass
    # them into the ranking model. To do so, we need to reshape the user
    # embeddings to match the shape of movie embeddings.
    list_length = features["movie_title"].shape[1]
    user_embedding_repeated = tf.repeat(
        tf.expand_dims(user_embeddings, 1), [list_length], axis=1)

    # Once reshaped, we concatenate and pass into the dense layers to generate
    # predictions.
    concatenated_embeddings = tf.concat(
        [user_embedding_repeated, movie_embeddings], 2)
    
    return self.score_model(concatenated_embeddings)

  def compute_loss(self, features, training=False):
    labels = features.pop("user_rating")
    scores = self(features)
    return self.task(labels=labels, predictions=tf.squeeze(scores, axis=-1),)


### Training the models
epochs = 1

cached_train = train.shuffle(100_000).batch(batch_size).cache()
cached_test = test.batch(batch_size).cache()

# # ### Mean squared error model
# #
# # This model is very similar to the model in [the basic ranking tutorial](basic_ranking). We train the model to minimize the mean squared error between the actual ratings and predicted ratings. Therefore, this loss is computed individually for each movie and the training is pointwise.
# mse_model = RankingModel(tf.keras.losses.MeanSquaredError())
# mse_model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
# mse_model.fit(cached_train, epochs=epochs, verbose=1)
#
# # ### Pairwise hinge loss model
# # By minimizing the pairwise hinge loss, the model tries to maximize the difference between the model's predictions for a highly rated item and a low rated item: the bigger that difference is, the lower the model loss. However, once the difference is large enough, the loss becomes zero, stopping the model from further optimizing this particular pair and letting it focus on other pairs that are incorrectly ranked
# # This loss is not computed for individual movies, but rather for pairs of movies. Hence the training using this loss is pairwise.
#
# hinge_model = RankingModel(tfr.keras.losses.PairwiseHingeLoss())
# hinge_model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
# hinge_model.fit(cached_train, epochs=epochs, verbose=1)
#

# ### Listwise model
# The `ListMLE` loss from TensorFlow Ranking expresses list maximum likelihood estimation. To calculate the ListMLE loss, we first use the user ratings to generate an optimal ranking. We then calculate the likelihood of each candidate being out-ranked by any item below it in the optimal ranking using the predicted scores. The model tries to minimize such likelihood to ensure highly rated candidates are not out-ranked by low rated candidates. You can learn more about the details of ListMLE in section 2.2 of the paper [Position-aware ListMLE: A Sequential Learning Process](http://auai.org/uai2014/proceedings/individuals/164.pdf).
# Note that since the likelihood is computed with respect to a candidate and all candidates below it in the optimal ranking, the loss is not pairwise but listwise. Hence the training uses list optimization.

listwise_model = RankingModel(tfr.keras.losses.ListMLELoss())
listwise_model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
listwise_model.fit(cached_train, epochs=epochs, verbose=1)


# ## Comparing the models

# mse_model_result = mse_model.evaluate(cached_test, return_dict=True)
# print("NDCG of the MSE Model: {:.4f}".format(mse_model_result["ndcg_metric"]))
#
# hinge_model_result = hinge_model.evaluate(cached_test, return_dict=True)
# print("NDCG of the pairwise hinge loss model: {:.4f}".format(hinge_model_result["ndcg_metric"]))

listwise_model_result = listwise_model.evaluate(cached_test, return_dict=True)
print("NDCG of the ListMLE model: {:.4f}".format(listwise_model_result["ndcg_metric"]))

# Of the three models, the model trained using ListMLE has the highest NDCG metric. This result shows how listwise optimization can be used to train ranking models and can potentially produce models that perform better than models optimized in a pointwise or pairwise fashion.
