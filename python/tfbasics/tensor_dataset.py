import numpy as np
import tensorflow as tf

# tensor
rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
print(rank_1_tensor)
print(rank_1_tensor.shape)
reshaped = tf.reshape(rank_1_tensor, [1, 3])
print(reshaped)
print(reshaped[0, 1].numpy())

# Dataset
# how to see it
# how to create
# scale, tf.data.Iterator does not need scale

dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
for e in dataset.take(2):
    print(e)
    print(e.numpy())

x = np.random.rand(10, 2)
dataset2 = tf.data.Dataset.from_tensor_slices(x)
for e in dataset2.take(5):
    print(e.numpy())

