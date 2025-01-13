
#!/usr/bin/env python
# coding: utf-8

# **First of all, set environment variables and initialize spark context:**

# In[1]:

from zoo.common.nncontext import *
sc = init_nncontext(init_spark_conf().setMaster("local[4]"))

from zoo.pipeline.api.keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(nb_words=10000)

print(max([max(sequence) for sequence in train_data]))

print(train_data[0:2])
print(train_labels[0:2])

# word_index is a dictionary mapping words to an integer index
word_index = imdb.get_word_index()
print(word_index['hello'])
print(len(word_index))
# We reverse it, mapping integer indices to words
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# We decode the review; note that our indices were offset by 3
# because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print(decoded_review)

import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results

# Our vectorized training data
x_train = vectorize_sequences(train_data)
# Our vectorized test data
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

print(x_train[0:2])
print(y_train[0:2])

from zoo.pipeline.api.keras import models
from zoo.pipeline.api.keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]



import time
dir_name = '3-5 ' + str(time.ctime())
model.set_tensorboard('./', dir_name)
model.fit(partial_x_train,
          partial_y_train,
          nb_epoch=20,
          batch_size=512,
          validation_data=(x_val, y_val))


train_loss = np.array(model.get_train_summary('Loss'))
val_loss = np.array(model.get_validation_summary('Loss'))

import matplotlib.pyplot as plt
plt.plot(train_loss[:,0],train_loss[:,1],label='train loss')
plt.plot(val_loss[:,0],val_loss[:,1],label='validation loss',color='green')
plt.title('Training and validation loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.show()


model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, nb_epoch=4, batch_size=512)


# _INFO - Trained 512 records in 0.023978868 seconds. Throughput is 21352.133 records/second. Loss is 0.108611815._

# In[15]:


results = model.evaluate(x_test, y_test)
print(results)


prediction = model.predict(x_test)
result = prediction.collect()
result


model.fit(partial_x_train,
          partial_y_train,
          nb_epoch=20,
          batch_size=512,
          validation_data=(x_val, y_val))


from bigdl.util.common import to_sample_rdd

train = to_sample_rdd(partial_x_train, partial_y_train)
val = to_sample_rdd(x_val, y_val)

model.fit(train, None,
          nb_epoch=20,
          batch_size=512,
          validation_data=val)


# This code zip the training data and label into RDD. The reason why it works is that every time when `fit` method takes `ndarray` as input, it transforms the `ndarray` to RDD and some memory is taken for cache in this process. And in this notebook, we use the same dataset as input repeatedly. If we call this operation only once and reuse the RDD afterwards, all the subsequential memory use would be saved.
