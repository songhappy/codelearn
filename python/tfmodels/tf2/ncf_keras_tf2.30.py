import keras
from tensorflow.keras import initializers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, Dense, Flatten, Concatenate, Multiply
import time
from bigdl.dataset import movielens
import numpy as np
import tensorflow as tf
import random

# keras 2.4.3, tensorflow 2.3
def init_normal(shape, name=None):
    return initializers.normal(shape, scale=0.01, name=name)

def get_model(num_users, num_items, layers=[20, 10], reg_layers=[0, 0]):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers)  # Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    print(user_input)
    MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=int(layers[0] / 2),
                                   embeddings_initializer='uniform',
                                   embeddings_regularizer=None,
                                   input_length=1)(user_input)
    MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=int(layers[0] / 2),
                                   embeddings_initializer='uniform',
                                   embeddings_regularizer=None,
                                   input_length=1)(item_input)

    # Crucial to flatten an embedding vector!
    # user_embed = MLP_Embedding_User(user_input)
    user_latent = Flatten()(MLP_Embedding_User)
    item_latent = Flatten()(MLP_Embedding_Item)

    # The 0-th layer is the concatenation of embedding layers
    mlp_linear = Concatenate()([user_latent, item_latent])
    # MLP layers
    for idx in range(1, num_layer):
        layer = Dense(layers[idx],  activation='relu',
                      name='layer%d' % idx)
        mlp_linear = layer(mlp_linear)

    # Final prediction layer
    prediction = Dense(5, activation='softmax', name='prediction')(mlp_linear)

    model = Model(inputs=[user_input, item_input],
                  outputs=prediction)
    return model



def get_full_model(num_users, num_items, class_num, layers=[20, 10], include_mf = True, mf_embed = 20):
    num_layer = len(layers)  # Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    print(user_input)
    mlp_embed_user = Embedding(input_dim=num_users, output_dim=int(layers[0] / 2),
                                   embeddings_initializer='uniform',
                                   embeddings_regularizer=None,
                                   input_length=1)(user_input)
    mlp_embed_item = Embedding(input_dim=num_items, output_dim=int(layers[0] / 2),
                                   embeddings_initializer='uniform',
                                   embeddings_regularizer=None,
                                   input_length=1)(item_input)

    # Crucial to flatten an embedding vector!
    # user_embed = MLP_Embedding_User(user_input)
    user_latent = Flatten()(mlp_embed_user)
    item_latent = Flatten()(mlp_embed_item)

    # The 0-th layer is the concatenation of embedding layers
    mlf_latent = Concatenate()([user_latent, item_latent])
    # MLP layers
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], activation='relu',
                      name='layer%d' % idx)
        mlf_latent = layer(mlf_latent)

    if (include_mf):
        mf_embed_user = Embedding(input_dim=num_users,
                                  output_dim=mf_embed,
                                  embeddings_initializer='uniform',
                                  input_length=1)(user_input)
        mf_embed_item = Embedding(input_dim=num_users,
                                  output_dim=mf_embed,
                                  embeddings_initializer='uniform',
                                  input_length=1)(item_input)
        mf_user_flatten = Flatten()(mf_embed_user)
        mf_item_flatten = Flatten()(mf_embed_item)
        mf_latent = Multiply()([mf_user_flatten, mf_item_flatten])
        concated_model = Concatenate()([mlf_latent, mf_latent])
        linear_last = Dense(class_num, activation="softmax")(concated_model)
    else:
        linear_last = Dense(class_num, activation='softmax', name='prediction')(mlf_latent)

    model = Model(inputs=[user_input, item_input],
                  outputs=linear_last)
    return model

if __name__ == '__main__':
    epochs = 10
    batch_size = 1024
    movielens_data = movielens.get_id_ratings("/Users/guoqiong/intelWork/data/movielens/")
    print(movielens_data[:3])
    random.shuffle(movielens_data)
    print(len(movielens_data))
    train = movielens_data[:800000]
    test = movielens_data[800000:]

    user_input = train[:,0]
    item_input = train[:,1]
    labels = train[:,2]
    test_user_input = test[:, 0]
    test_item_input = test[:, 1]
    test_label = test[:, 2]
    def tozerobased(l):
        return l - 1

    user_input = np.array(list(map(lambda x: int(x.reshape([1])), user_input)))
    item_input = np.array(list(map(lambda x: int(x.reshape([1])), item_input)))
    labels = np.array(list(map(tozerobased, labels)))
    test_user_input = np.array(list(map(lambda x: int(x.reshape([1])), test_user_input)))
    test_item_input = np.array(list(map(lambda x: int(x.reshape([1])), test_item_input)))
    test_labels = np.array(list(map(tozerobased, test_label)))

    print(type(labels))
    print(np.array(user_input)[:10])
    print(np.array(item_input)[:10])
    print(labels[:10])
    import sys
    #sys.exit()
    num_users = max(movielens_data[:,0])
    num_items = max(movielens_data[:,1])
    model = get_full_model(num_users + 1, num_items + 1, 5)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])
    # for binary, keras.metrics.AUC(curve='ROC'), keras.metrics.AUC(curve='PR')
    for epoch in range(epochs):
        t1 = time.time()
        # Generate training instances

        # Training
        hist = model.fit([np.array(user_input), np.array(item_input)],  # input
                         labels,  # labels
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        score = model.evaluate([test_user_input, test_item_input], test_labels, verbose=0)
        print(score)

    test = model.predict([np.array([10]), np.array([10])])
    print(test)

    filepath = "/Users/guoqiong/intelWork/git/tensorflow/tf-models/models/ncf23"

    tf.saved_model.save(model, filepath)
    loaded = tf.saved_model.load(filepath)
    print(list(loaded.signatures.keys()))
    infer = loaded.signatures["serving_default"]
    print(infer.structured_outputs)
    test_input = [tf.TensorSpec(shape=(None, 1), dtype=tf.int32, name='user_input'), tf.TensorSpec(shape=(None, 1), dtype=tf.int32, name='item_input')]
    test_input = [np.array([1]), np.array([1])]

    loaded1 = tf.keras.models.load_model(filepath)
    test_out = loaded1.predict(test_input)
    print(test_out)
    print(np.argmax(test_out, axis=1))

    # with tf.Session(graph=tf.Graph()) as sess:
    #     tf.saved_model.loader.load(sess, ["serve"], filepath)
    #     graph = tf.get_default_graph()
    #     print(graph.get_operations())
    #     sess.run('myOutput:0',
    #              feed_dict={'myInput:0':test_input})