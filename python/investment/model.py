from keras.layers import Input, Dense, LSTM, Conv1D, Dropout, Bidirectional, Multiply
from keras.models import Model
# from attention_utils import get_activations
from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
from utils import *
import numpy as np
import xgboost as xgb
import keras as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPool1D, Activation, multiply, Input, LSTM
from tensorflow.keras.optimizers import Adam


def attention_one(x_train, y_train, x_test, y_test, features):
    shape = features.shape
    features = features.reshape(-1, shape[0], shape[1])
    input_shape = (x_train.shape[1], x_train.shape[2])
    inputs = Input(shape=input_shape)

    cnn = Conv1D(filters=64, kernel_size=1, activation='relu')(inputs)  # padding = 'same'

    # Define the attention mechanism layer
    def attention_3d_block(inputs):
        hidden_states = inputs
        hidden_size = int(hidden_states.shape[2])
        score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(
            hidden_states)
        score_activation = Activation('tanh')(score_first_part)
        attention_weights = Dense(1, name='attention_weight_vec')(score_activation)
        attention_weights = Activation('softmax')(attention_weights)
        context_vector = multiply([hidden_states, attention_weights])
        return context_vector

    # Build the model

    lstm_out = LSTM(64, return_sequences=True)(cnn)
    attention_mul = attention_3d_block(lstm_out)
    attention_flatten = Flatten()(attention_mul)
    output = Dense(1)(attention_flatten)
    model = Model(inputs=inputs, outputs=output)

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=False)

    # Evaluate the model
    mse = model.evaluate(x_test, y_test)
    # print('Mean Squared Error:', mse)

    # Make predictions
    predictions = model.predict(features)
    return predictions

def cnn_one(x_train, y_train, x_test, y_test, features):

    shape = features.shape
    features = features.reshape(-1, shape[0], shape[1])

    model = Sequential()

    model.add(Conv1D(filters=32, kernel_size=(5), padding='Same',
                     activation='relu', input_shape=shape))
    model.add(Conv1D(filters=32, kernel_size=(5), padding='Same',
                     activation='relu'))
    model.add(MaxPool1D(pool_size=(5)))

    model.add(Conv1D(filters=64, kernel_size=(3), padding='Same',
                     activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=(3), padding='Same',
                     activation='relu'))
    model.add(MaxPool1D(pool_size=(3), strides=(2)))

    model.add(Flatten())
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss="mse", metrics=["mse"])
    model.fit(x_train, y_train, epochs=10, verbose=False)
    predictions = model(features)
    return predictions.numpy()

def attention_3d_block_merge(inputs,single_attention_vector = False):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = inputs
    # a = Permute((2, 1))(inputs)
    # a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(input_dim, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((1, 2), name='attention_vec')(a)

    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul

def attention_3d_block(inputs, single_attention_vector=False):
    # inputs.shape = (batch_size, time_steps, input_dim)
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1))(a)
    # element-wise
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

def attention_model(INPUT_DIMS = 13,TIME_STEPS = 20,lstm_units = 64):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIMS))

    x = Conv1D(filters=64, kernel_size=1, activation='relu')(inputs)  # padding = 'same'
    x = Dropout(0.3)(x)

    # lstm_out = Bidirectional(LSTM(lstm_units, activation='relu'), name='bilstm')(x)
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    lstm_out = Dropout(0.3)(lstm_out)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)

    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model

def PredictWithData(data,data_yuan,name,modelname,INPUT_DIMS = 13,TIME_STEPS = 20):
    print(data.columns)
    yindex = data.columns.get_loc(name)
    data = np.array(data, dtype='float64')
    data, normalize = NormalizeMult(data)
    data_y = data[:, yindex]
    data_y = data_y.reshape(data_y.shape[0], 1)

    testX, _ = create_dataset(data)
    _, testY = create_dataset(data_y)
    print("testX Y shape is:", testX.shape, testY.shape)
    if len(testY.shape) == 1:
        testY = testY.reshape(-1, 1)

    model = attention_model(INPUT_DIMS)
    model.load_weights(modelname)
    model.summary()
    y_hat =  model.predict(testX)
    testY, y_hat = xgb_scheduler(data_yuan, y_hat)
    return y_hat, testY

def lstm(model_type,X_train,yuan_X_train):
    if model_type == 1:
        # single-layer LSTM
        model = Sequential()
        model.add(LSTM(units=50, activation='relu',
                    input_shape=(X_train.shape[1], 1)))
        model.add(Dense(units=1))
        yuan_model = Sequential()
        yuan_model.add(LSTM(units=50, activation='relu',
                    input_shape=(yuan_X_train.shape[1], 5)))
        yuan_model.add(Dense(units=5))
    if model_type == 2:
        # multi-layer LSTM
        model = Sequential()
        model.add(LSTM(units=50, activation='relu', return_sequences=True,
                    input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(units=50, activation='relu'))
        model.add(Dense(1))

        yuan_model = Sequential()
        yuan_model.add(LSTM(units=50, activation='relu', return_sequences=True,
                    input_shape=(yuan_X_train.shape[1], 5)))
        yuan_model.add(LSTM(units=50, activation='relu'))
        yuan_model.add(Dense(5))
    if model_type == 3:
        # BiLSTM
        model = Sequential()
        model.add(Bidirectional(LSTM(50, activation='relu'),
                                input_shape=(X_train.shape[1], 1)))
        model.add(Dense(1))

        yuan_model = Sequential()
        yuan_model.add(Bidirectional(LSTM(50, activation='relu'),
                                    input_shape=(yuan_X_train.shape[1], 5)))
        yuan_model.add(Dense(5))

    return model,yuan_model

def xgb_scheduler(data,y_hat):
    close = data.pop('close')
    data.insert(5, 'close', close)
    train, test = prepare_data(data, n_test=len(y_hat), n_in=6, n_out=1)
    testY, y_hat2 = walk_forward_validation(train, test)
    return testY, y_hat2

def xgboost_forecast(train, testX):
    # transform list into array
    train = np.asarray(train)
    # print('train', train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # print('trainX', trainX, 'trainy', trainy)
    # fit model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=20)
    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict(np.asarray([testX]))
    return yhat[0]

def walk_forward_validation(train, test):
    predictions = list()
    train = train.values
    history = [x for x in train]
    # print('history', history)
    for i in range(len(test)):
        testX, testy = test.iloc[i, :-1], test.iloc[i, -1]
        # print('i', i, testX, testy)
        yhat = xgboost_forecast(history, testX)
        predictions.append(yhat)
        history.append(test.iloc[i, :])
        print(i+1, '>expected=%.6f, predicted=%.6f' % (testy, yhat))
    return test.iloc[:, -1],predictions