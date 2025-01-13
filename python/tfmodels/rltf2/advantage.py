from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import mnist
from keras.utils import np_utils

import numpy as np
from sklearn.linear_model import LogisticRegression
def build_logistic_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(output_dim, input_dim=input_dim, activation='softmax'))

    return model

x = [[0, i] for i in range(50)]
y = [j for j in range(50)]
for j in range(50):
    if j // 2 == 0:
        y[j] = [1,0]
    else:
        y[j] = [0, 1*1]

x = np.array(x)
y = np.array(y)

model = build_logistic_model(input_dim=2, output_dim=2)

# compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x, y, batch_size=32, epochs=10)

for i in range(10):
    y1 = model.predict(np.array([0, i]).reshape(-1,2))
    print( y1)
