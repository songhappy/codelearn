# from
# https://github.com/tensorflow/docs/blob/master/site/en/guide/saved_model.ipynb
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf
import tensorflow as tf

tf.enable_eager_execution()

tmpdir="/Users/guoqiong/intelWork/git/tensorflow/tf-models/models/linear"
input_column = tf.feature_column.numeric_column("x")
estimator = tf.estimator.LinearClassifier(feature_columns=[input_column])

def input_fn():
  return tf.data.Dataset.from_tensor_slices(
    ({"x": [1., 2., 3., 4.]}, [1, 1, 0, 0])).repeat(200).shuffle(64).batch(16)
estimator.train(input_fn)

serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
  tf.feature_column.make_parse_example_spec([input_column]))
estimator_base_path = os.path.join(tmpdir, 'from_estimator')
estimator_path = estimator.export_saved_model(estimator_base_path, serving_input_fn)

#model_input="/Users/guoqiong/intelWork/git/tensorflow/tf-models/models/linear/from_estimator/1597104456"
imported = tf.compat.v2.saved_model.load(estimator_path, tags=None)

def predict(x):
  example = tf.train.Example()
  example.features.feature["x"].float_list.value.extend([x])
  print(example)
  print(tf.constant([example.SerializeToString()]))
  return imported.signatures["predict"](
    examples=tf.constant([example.SerializeToString()]))

print(predict(1.5))
print(predict(3.5))

