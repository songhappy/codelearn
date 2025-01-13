# Build the model of a logistic classifier
# y to_categorical(y)

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import sys
def build_logistic_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(output_dim, input_dim=input_dim, activation='softmax'))

    return model

batch_size = 128
nb_classes = 10
nb_epoch = 20
input_dim = 784

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(type(X_train))
print(type(X_train[0][0][0]))
print(X_train.shape)

X_train = X_train.reshape(60000, input_dim)
X_test = X_test.reshape(10000, input_dim)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = to_categorical(y_train, nb_classes)
Y_test = to_categorical(y_test, nb_classes)
# Y_train = y_train
# Y_test = y_test
print(Y_train.shape)
print(X_train.shape)
print(Y_train[0])

model = build_logistic_model(input_dim, nb_classes)

model.summary()

# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC'), tf.keras.metrics.AUC(curve='PR')])
history = model.fit(X_train, Y_train,
                    batch_size=batch_size, epochs=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))


score = model.evaluate(X_test, Y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('Test metircs:', score[:])

# save model as json and yaml
json_string = model.to_json()
open('mnist_Logistic_model.json', 'w').write(json_string)
yaml_string = model.to_yaml()
open('mnist_Logistic_model.yaml', 'w').write(yaml_string)

# save the weights in h5 format
model.save_weights('mnist_Logistic_wts.h5', overwrite=True)
model.save('mnist_Logistic.h5', overwrite=True)

# to read a saved model and weights
# model = model_from_json(open('my_model_architecture.json').read())
# model = model_from_yaml(open('my_model_architecture.yaml').read())

loaded2 = build_logistic_model(input_dim, nb_classes)
loaded2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC'), tf.keras.metrics.AUC(curve='PR')])

loaded1 = tf.keras.models.load_model("mnist_Logistic.h5")
loaded2.load_weights('mnist_Logistic_wts.h5')

score1 = loaded1.evaluate(X_test, Y_test, verbose=0)
score2 = loaded2.evaluate(X_test, Y_test, verbose=0)

print(score)
print(score1)
print(score2)

predictions = loaded2.predict(X_test)
prediction_classes = np.argmax(predictions, axis=1)
for i in range(5):
    print(prediction_classes[i])
    print(predictions[i])