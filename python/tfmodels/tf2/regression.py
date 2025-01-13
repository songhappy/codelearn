#!/usr/bin/env python
# coding: utf-8

# ##### Copyright 2018 The TensorFlow Authors.

# In[ ]:


#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# In[ ]:


#@title MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# # Basic regression: Predict fuel efficiency

# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/tutorials/keras/regression"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/regression.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/regression.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
#   <td>
#     <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/tutorials/keras/regression.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
#   </td>
# </table>

# In a *regression* problem, the aim is to predict the output of a continuous value, like a price or a probability. Contrast this with a *classification* problem, where the aim is to select a class from a list of classes (for example, where a picture contains an apple or an orange, recognizing which fruit is in the picture).
# 
# This notebook uses the classic [Auto MPG](https://archive.ics.uci.edu/ml/datasets/auto+mpg) Dataset and builds a model to predict the fuel efficiency of late-1970s and early 1980s automobiles. To do this, provide the model with a description of many automobiles from that time period. This description includes attributes like: cylinders, displacement, horsepower, and weight.
# 
# This example uses the `tf.keras` API, see [this guide](https://www.tensorflow.org/guide/keras) for details.

# In[ ]:


# Use seaborn for pairplot
get_ipython().system('pip install -q seaborn')


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)


# In[ ]:


import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

print(tf.__version__)


# ## The Auto MPG dataset
# 
# The dataset is available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/).
# 

# ### Get the data
# First download and import the dataset using pandas:

# In[ ]:


url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)


# In[ ]:


dataset = raw_dataset.copy()
dataset.tail()


# ### Clean the data
# 
# The dataset contains a few unknown values.

# In[ ]:


dataset.isna().sum()


# Drop those rows to keep this initial tutorial simple.

# In[ ]:


dataset = dataset.dropna()


# The `"Origin"` column is really categorical, not numeric. So convert that to a one-hot with [pd.get_dummies](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html):
# 
# Note: You can set up the `keras.Model` to do this kind of transformation for you. That's beyond the scope of this tutorial. See the [preprocessing layers](../structured_data/preprocessing_layers.ipynb) or [Loading CSV data](../load_data/csv.ipynb) tutorials for examples.

# In[ ]:


dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})


# In[ ]:


dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
dataset.tail()


# ### Split the data into train and test
# 
# Now split the dataset into a training set and a test set.
# 
# Use the test set in the final evaluation of your models.

# In[ ]:


train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)


# ### Inspect the data
# 
# Have a quick look at the joint distribution of a few pairs of columns from the training set.
# 
# Looking at the top row it should be clear that the fuel efficiency (MPG) is a function of all the other parameters. Looking at the other rows it should be clear that they are functions of each other.

# In[ ]:


sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')


# Also look at the overall statistics, note how each feature covers a very different range:

# In[ ]:


train_dataset.describe().transpose()


# ### Split features from labels
# 
# Separate the target value, the "label", from the features. This label is the value that you will train the model to predict.

# In[ ]:


train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')


# ## Normalization
# 
# In the table of statistics it's easy to see how different the ranges of each feature are.

# In[ ]:


train_dataset.describe().transpose()[['mean', 'std']]


# It is good practice to normalize features that use different scales and ranges. 
# 
# One reason this is important is because the features are multiplied by the model weights. So the scale of the outputs and the scale of the gradients are affected by the scale of the inputs. 
# 
# Although a model *might* converge without feature normalization, normalization makes training much more stable.
# 
# Note: There is no advantage to normalizing the one-hot features, it is done here for simplicity. For more details on how to use the preprocessing layers, refer the [Working with preprocessing layers](https://www.tensorflow.org/guide/keras/preprocessing_layers) guide and the [Classify structured data using Keras preprocessing layers](https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers) tutorial.

# ### The Normalization layer
# The `preprocessing.Normalization` layer is a clean and simple way to build that preprocessing into your model.
# 
# The first step is to create the layer:

# In[ ]:


normalizer = preprocessing.Normalization(axis=-1)


# Then `.adapt()` it to the data:

# In[ ]:


normalizer.adapt(np.array(train_features))


# This calculates the mean and variance, and stores them in the layer. 

# In[ ]:


print(normalizer.mean.numpy())


# When the layer is called it returns the input data, with each feature independently normalized:

# In[ ]:


first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())


# ## Linear regression
# 
# Before building a DNN model, start with a linear regression.

# ### One Variable
# 
# Start with a single-variable linear regression, to predict `MPG` from `Horsepower`.
# 
# Training a model with `tf.keras` typically starts by defining the model architecture.
# 
# In this case use a `keras.Sequential` model. This model represents a sequence of steps. In this case there are two steps:
# 
# * Normalize the input `horsepower`.
# * Apply a linear transformation ($y = mx+b$) to produce 1 output using `layers.Dense`.
# 
# The number of _inputs_ can either be set by the `input_shape` argument, or automatically when the model is run for the first time.

# First create the horsepower `Normalization` layer:

# In[ ]:


horsepower = np.array(train_features['Horsepower'])

horsepower_normalizer = preprocessing.Normalization(input_shape=[1,], axis=None)
horsepower_normalizer.adapt(horsepower)


# Build the sequential model:

# In[ ]:


horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(units=1)
])

horsepower_model.summary()


# This model will predict `MPG` from `Horsepower`.
# 
# Run the untrained model on the first 10 horse-power values. The output won't be good, but you'll see that it has the expected shape, `(10,1)`:

# In[ ]:


horsepower_model.predict(horsepower[:10])


# Once the model is built, configure the training procedure using the `Model.compile()` method. The most important arguments to compile are the `loss` and the `optimizer` since these define what will be optimized (`mean_absolute_error`) and how (using the `optimizers.Adam`).

# In[ ]:


horsepower_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')


# Once the training is configured, use `Model.fit()` to execute the training:

# In[ ]:


get_ipython().run_cell_magic('time', '', "history = horsepower_model.fit(\n    train_features['Horsepower'], train_labels,\n    epochs=100,\n    # suppress logging\n    verbose=0,\n    # Calculate validation results on 20% of the training data\n    validation_split = 0.2)")


# Visualize the model's training progress using the stats stored in the `history` object.

# In[ ]:


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


# In[ ]:


def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)


# In[ ]:


plot_loss(history)


# Collect the results on the test set, for later:

# In[ ]:


test_results = {}

test_results['horsepower_model'] = horsepower_model.evaluate(
    test_features['Horsepower'],
    test_labels, verbose=0)


# Since this is a single variable regression it's easy to look at the model's predictions as a function of the input:

# In[ ]:


x = tf.linspace(0.0, 250, 251)
y = horsepower_model.predict(x)


# In[ ]:


def plot_horsepower(x, y):
  plt.scatter(train_features['Horsepower'], train_labels, label='Data')
  plt.plot(x, y, color='k', label='Predictions')
  plt.xlabel('Horsepower')
  plt.ylabel('MPG')
  plt.legend()


# In[ ]:


plot_horsepower(x,y)


# ### Multiple inputs

# You can use an almost identical setup to make predictions based on multiple inputs. This model still does the same $y = mx+b$ except that $m$ is a matrix and $b$ is a vector.
# 
# This time use the `Normalization` layer that was adapted to the whole dataset.

# In[ ]:


linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])


# When you call this model on a batch of inputs, it produces `units=1` outputs for each example.

# In[ ]:


linear_model.predict(train_features[:10])


# When you call the model it's weight matrices will be built. Now you can see that the `kernel` (the $m$ in $y=mx+b$) has a shape of `(9,1)`.

# In[ ]:


linear_model.layers[1].kernel


# Use the same `compile` and `fit` calls as for the single input `horsepower` model:

# In[ ]:


linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history = linear_model.fit(\n    train_features, train_labels, \n    epochs=100,\n    # suppress logging\n    verbose=0,\n    # Calculate validation results on 20% of the training data\n    validation_split = 0.2)')


# Using all the inputs achieves a much lower training and validation error than the `horsepower` model: 

# In[ ]:


plot_loss(history)


# Collect the results on the test set, for later:

# In[ ]:


test_results['linear_model'] = linear_model.evaluate(
    test_features, test_labels, verbose=0)


# ## A DNN regression

# The previous section implemented linear models for single and multiple inputs.
# 
# This section implements single-input and multiple-input DNN models. The code is basically the same except the model is expanded to include some "hidden"  non-linear layers. The name "hidden" here just means not directly connected to the inputs or outputs.

# These models will contain a few more layers than the linear model:
# 
# * The normalization layer.
# * Two hidden, nonlinear, `Dense` layers using the `relu` nonlinearity.
# * A linear single-output layer.
# 
# Both will use the same training procedure so the `compile` method is included in the `build_and_compile_model` function below.

# In[ ]:


def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model


# ### One variable

# Start with a DNN model for a single input: "Horsepower"

# In[ ]:


dnn_horsepower_model = build_and_compile_model(horsepower_normalizer)


# This model has quite a few more trainable parameters than the linear models.

# In[ ]:


dnn_horsepower_model.summary()


# Train the model:

# In[ ]:


get_ipython().run_cell_magic('time', '', "history = dnn_horsepower_model.fit(\n    train_features['Horsepower'], train_labels,\n    validation_split=0.2,\n    verbose=0, epochs=100)")


# This model does slightly better than the linear-horsepower model.

# In[ ]:


plot_loss(history)


# If you plot the predictions as a function of `Horsepower`, you'll see how this model takes advantage of the nonlinearity provided by the hidden layers:

# In[ ]:


x = tf.linspace(0.0, 250, 251)
y = dnn_horsepower_model.predict(x)


# In[ ]:


plot_horsepower(x, y)


# Collect the results on the test set, for later:

# In[ ]:


test_results['dnn_horsepower_model'] = dnn_horsepower_model.evaluate(
    test_features['Horsepower'], test_labels,
    verbose=0)


# ### Full model

# If you repeat this process using all the inputs it slightly improves the performance on the validation dataset.

# In[ ]:


dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history = dnn_model.fit(\n    train_features, train_labels,\n    validation_split=0.2,\n    verbose=0, epochs=100)')


# In[ ]:


plot_loss(history)


# Collect the results on the test set:

# In[ ]:


test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)


# ## Performance

# Now that all the models are trained check the test-set performance and see how they did:

# In[ ]:


pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T


# These results match the validation error seen during training.

# ### Make predictions
# 
# Finally, predict have a look at the errors made by the model when making predictions on the test set:

# In[ ]:


test_predictions = dnn_model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# It looks like the model predicts reasonably well. 
# 
# Now take a look at the error distribution:

# In[ ]:


error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
_ = plt.ylabel('Count')


# If you're happy with the model save it for later use:

# In[ ]:


dnn_model.save('dnn_model')


# If you reload the model, it gives identical output:

# In[ ]:


reloaded = tf.keras.models.load_model('dnn_model')

test_results['reloaded'] = reloaded.evaluate(
    test_features, test_labels, verbose=0)


# In[ ]:


pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T


# ## Conclusion
# 
# This notebook introduced a few techniques to handle a regression problem. Here are a few more tips that may help:
# 
# * [Mean Squared Error (MSE)](https://www.tensorflow.org/api_docs/python/tf/losses/MeanSquaredError) and [Mean Absolute Error (MAE)](https://www.tensorflow.org/api_docs/python/tf/losses/MeanAbsoluteError) are common loss functions used for regression problems. Mean Absolute Error is less sensitive to outliers. Different loss functions are used for classification problems.
# * Similarly, evaluation metrics used for regression differ from classification.
# * When numeric input data features have values with different ranges, each feature should be scaled independently to the same range.
# * Overfitting is a common problem for DNN models, it wasn't a problem for this tutorial. See the [overfit and underfit](overfit_and_underfit.ipynb) tutorial for more help with this.
# 
