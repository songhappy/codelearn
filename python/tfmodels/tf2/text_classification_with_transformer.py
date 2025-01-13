"""
Title: Text classification with Transformer
Author: [Apoorv Nandan](https://twitter.com/NandanApoorv)
Date created: 2020/05/10
Last modified: 2020/05/10
Description: Implement a Transformer block as a Keras layer and use it for text classification.
"""
"""
## Setup
"""
## modify the reading text part into my style

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
"""
## Download and prepare dataset
"""
vocab_size = 20000  # Only consider the top 20k words
maxlen = 200  # Only consider the first 200 words of each movie review
embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer
(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=vocab_size)
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")
for text_batch in x_train.take(1):
    print(text_batch.numpy()[:3])
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

from models import build_transformer_model
model = build_transformer_model(maxlen=maxlen,
                                vocab_size=vocab_size,
                                embed_dim=embed_dim,
                                num_heads=num_heads,
                                ff_dim=ff_dim,
                                num_labels=2)

"""
## Create classifier model using transformer layer
Transformer layer outputs one vector for each time step of our input sequence.
Here, we take the mean across all time steps and
use a feed forward network on top of it to classify text.
"""



"""
## Train and Evaluate
"""
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(
    x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val))