# from https://github.com/tensorflow/recommenders/blob/main/docs/examples/basic_ranking.ipynb
#!/usr/bin/env python
# Real-world recommender systems are often composed of two stages:
# 
# 1. The retrieval stage is responsible for selecting an initial set of hundreds of candidates from all possible candidates. The main objective of this model is to efficiently weed out all candidates that the user is not interested in. Because the retrieval model may be dealing with millions of candidates, it has to be computationally efficient.
# 2. The ranking stage takes the outputs of the retrieval model and fine-tunes them to select the best possible handful of recommendations. Its task is to narrow down the set of items the user may be interested in to a shortlist of likely candidates.
# 
# We're going to focus on the second stage, ranking. If you are interested in the retrieval stage, have a look at our [retrieval](basic_retrieval) tutorial.
# 
# In this tutorial, we're going to:
# 
# 1. Get our data and split it into a training and test set.
# 2. Implement a ranking model.
# 3. Fit and evaluate it.

import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import sys
# We're going to use the same data as the [retrieval](basic_retrieval) tutorial. This time, we're also going to keep the ratings: these are the objectives we are trying to predict.


ratings = tfds.load("movielens/100k-ratings", split="train")
ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
    "user_rating": x["user_rating"]
})

# As before, we'll split the data by putting 80% of the ratings in the train set, and 20% in the test set.
tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)
for e in test.take(2):
    print(e)
    print(e.keys())

# Let's also figure out unique user ids and movie titles present in the data.
# This is important because we need to be able to map the raw values of our categorical features to embedding vectors in our models. To do that, we need a vocabulary that maps a raw feature value to an integer in a contiguous range: this allows us to look up the corresponding embeddings in our embedding tables.

movie_titles = ratings.batch(1_000_000).map(lambda x: x["movie_title"])
user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))
print(unique_movie_titles)
print(unique_user_ids)

# ## Implementing a model

# ### Architecture
# 
# Ranking models do not face the same efficiency constraints as retrieval models do, and so we have a little bit more freedom in our choice of architectures.
# A model composed of multiple stacked dense layers is a relatively common architecture for ranking tasks. We can implement it as follows:

class RankingModel(tf.keras.Model):

  def __init__(self):
    super().__init__()
    embedding_dimension = 32

    # Compute embeddings for users.
    self.user_embeddings = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_user_ids, mask_token=None),
      tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
    ])

    # Compute embeddings for movies.
    self.movie_embeddings = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_movie_titles, mask_token=None),
      tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
    ])

    # Compute predictions.
    self.ratings = tf.keras.Sequential([
      # Learn multiple dense layers.
      tf.keras.layers.Dense(256, activation="relu"),
      tf.keras.layers.Dense(64, activation="relu"),
      # Make rating predictions in the final layer.
      tf.keras.layers.Dense(1)
  ])
    
  def call(self, inputs):

    user_id, movie_title = inputs

    user_embedding = self.user_embeddings(user_id)
    movie_embedding = self.movie_embeddings(movie_title)

    return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))


# This model takes user ids and movie titles, and outputs a predicted rating:

# ### Loss and metrics
# The next component is the loss used to train our model. TFRS has several loss layers and tasks to make this easy.
# In this instance, we'll make use of the `Ranking` task object: a convenience wrapper that bundles together the loss function and metric computation.
# We'll use it together with the `MeanSquaredError` Keras loss in order to predict the ratings.

task = tfrs.tasks.Ranking(
  loss = tf.keras.losses.MeanSquaredError(),
  metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

# The task itself is a Keras layer that takes true and predicted as arguments, and returns the computed loss. We'll use that to implement the model's training loop.
# ### The full model
# We can now put it all together into a model. TFRS exposes a base model class (`tfrs.models.Model`) which streamlines bulding models: all we need to do is to set up the components in the `__init__` method, and implement the `compute_loss` method, taking in the raw features and returning a loss value.
# The base model will then take care of creating the appropriate training loop to fit our model.

# In[ ]:

class MovielensModel(tfrs.models.Model):

  def __init__(self):
    super().__init__()
    self.ranking_model: tf.keras.Model = RankingModel()
    self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
      loss=tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )

  def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
    return self.ranking_model((features["user_id"], features["movie_title"]))

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    labels = features.pop("user_rating")
    rating_predictions = self(features)
    # The task computes the loss and the metrics.
    return self.task(labels=labels, predictions=rating_predictions)


# ## Fitting and evaluating
# 
# After defining the model, we can use standard Keras fitting and evaluation routines to fit and evaluate the model.
# 
# Let's first instantiate the model.

model = MovielensModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))


# Then shuffle, batch, and cache the training and evaluation data.

cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()


# Then train the  model:

model.fit(cached_train, epochs=3)


# As the model trains, the loss is falling and the RMSE metric is improving.
# Finally, we can evaluate our model on the test set:


model.evaluate(cached_test, return_dict=True)


# The lower the RMSE metric, the more accurate our model is at predicting ratings.

# ## Testing the ranking model
# 
# Now we can test the ranking model by computing predictions for a set of movies and then rank these movies based on the predictions:
#

test_ratings = {}
test_movie_titles = ["M*A*S*H (1970)", "Dances with Wolves (1990)", "Speed (1994)"]
for movie_title in test_movie_titles:
  test_ratings[movie_title] = model({
      "user_id": np.array(["42"]),
      "movie_title": np.array([movie_title])
})

print("Ratings:")
for title, score in sorted(test_ratings.items(), key=lambda x: x[1], reverse=True):
  print(f"{title}: {score}")
