from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, Dense, Reshape, Flatten, Dropout, Concatenate, Multiply
import time
from bigdl.dataset import movielens
import numpy as np

# import keras from keras==1.2.2 has problem, so just use tensorflow.keras to avoid version imcompatibility.
def get_model(num_users, num_items, class_num, layers=[20, 10], include_mf = True, mf_embed = 20):
    num_layer = len(layers)  # Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    print(user_input)
    mlp_embed_user = Embedding(input_dim=num_users, output_dim=int(layers[0] / 2),
                               embeddings_initializer='uniform',
                               input_length=1)(user_input)
    mlp_embed_item = Embedding(input_dim=num_items, output_dim=int(layers[0] / 2),
                               embeddings_initializer='uniform',
                               input_length=1)(item_input)

    # Crucial to flatten an embedding vector!
    # user_embed = MLP_Embedding_User(user_input)
    user_latent = Flatten()(mlp_embed_user)
    item_latent = Flatten()(mlp_embed_item)

    # The 0-th layer is the concatenation of embedding layers
    # vector = merge([user_latent, item_latent], mode='concat')
    mlp_lalent = Concatenate()([user_latent, item_latent])
    # MLP layers
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], activation='relu',name='layer%d' % idx)
        mlp_lalent = layer(mlp_lalent)

    if (include_mf):
        mf_embed_user = Embedding(input_dim=num_users,
                                  output_dim=mf_embed,
                                  embeddings_initializer='uniform',
                                  name="user_embedding",
                                  input_length=1)(user_input)
        mf_embed_item = Embedding(input_dim=num_users,
                                  output_dim=mf_embed,
                                  name="item_embedding",
                                  embeddings_initializer='uniform',
                                  input_length=1)(item_input)
        mf_user_flatten = Flatten()(mf_embed_user)
        mf_item_flatten = Flatten()(mf_embed_item)
        mf_latent = Multiply()([mf_user_flatten, mf_item_flatten])
        concated_model = Concatenate()([mlp_lalent, mf_latent])
        prediction = Dense(class_num, activation="softmax")(concated_model)
    else:
        prediction = Dense(class_num, activation='softmax', name='prediction')(mlp_lalent)

    model = Model([user_input, item_input], prediction)
    return model

if __name__ == '__main__':
    epochs = 2
    batch_size = 1024
    movielens_data = movielens.get_id_ratings("/tmp/movielens/")
    print(movielens_data[:3])

    user_input = movielens_data[:,0]
    item_input = movielens_data[:,1]
    labels = movielens_data[:,2]
    # user_input = np.array(list(map(lambda x: x.reshape([1]), user_input)))
    # item_input = np.array(list(map(lambda x: x.reshape([1]), item_input)))
    def tozerobased(l):
        return l - 1
    labels = np.array(list(map(tozerobased, labels)))
    print(type(labels))
    print(np.array(user_input)[:10])
    print(np.array(item_input)[:10])
    print(labels[:10])
    import sys
    #sys.exit()
    num_users = max(user_input) + 1
    num_items = max(item_input) + 1
    model = get_model(num_users, num_items, 5)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])
    for epoch in range(epochs):
        t1 = time.time()
        # Generate training instances

        # Training
        hist = model.fit([np.array(user_input), np.array(item_input)],  # input
                         labels,  # labels
                         batch_size=batch_size, epochs=10, verbose=1, shuffle=True)

        test = model.predict([np.array([1]), np.array([1])])
        print(test)

    # for i, layer in enumerate(model.layers):
    #     print(i)
    #     print(layer.get_weights())

    layer_user = model.get_layer("user_embedding")
    layer_item = model.get_layer("item_embedding")

    print(layer_user)
    print(layer_user.get_weights()[0])
    print(type(layer_user.get_weights()[0]))
    print(layer_user.get_weights()[0].shape)
