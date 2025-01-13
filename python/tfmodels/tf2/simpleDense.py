from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras import layers

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

model1 = tf.keras.Sequential()
l1 = layers.Dense(64, activation='relu')
l2 = layers.Dense(64, activation='relu')
l3 = layers.Dense(10, activation='softmax')

# Adds a densely-connected layer with 64 units to the model:
model1.add(layers.Dense(64, activation='relu'))
# Add another:
model1.add(layers.Dense(64, activation='relu'))
# Add a softmax layer with 10 output units:
model1.add(layers.Dense(10, activation='softmax'))
model2 = tf.keras.Sequential([
# Adds a densely-connected layer with 64 units to the model:
layers.Dense(64, activation='relu', input_shape=(32,)),
# Add another:
layers.Dense(64, activation='relu'),
# Add a softmax layer with 10 output units:
layers.Dense(10, activation='softmax')])

model = tf.keras.Sequential([l1,l2,l3])

# Configure a model for mean-squared error regression.
model.compile(optimizer=tf.keras.optimizers.Adam(0.1),
              loss='mse',       # mean squared error
              metrics=['mae'])  # mean absolute error

import numpy as np

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

history = model.fit(data, labels, epochs=10, batch_size=32)
#print(l1.get_weights()[0].shape)
#print(l2.get_weights()[0].shape)
#print(l3.get_weights())

import matplotlib.pyplot as plt
plt.xlabel("epoch number")
plt.ylabel("loss magnitude")
plt.plot(history.history['loss'])
#plt.show()

# how to print the architecture and weights


celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheith = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

l0 = tf.keras.layers.Dense(1, input_shape=(1,))
model = tf.keras.Sequential([l0])
model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss='mse',       # mean squared error
              metrics=['mae'])  #
model.fit(celsius, fahrenheith, epochs=5000)
print(l0.get_weights())
