
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *



"""
## Implement a Transformer block as a layer
"""


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
"""
## Implement embedding layer
Two seperate embedding layers, one for tokens, one for token index (positions).
"""

class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

def build_transformer_model(maxlen, vocab_size, embed_dim, num_heads, ff_dim, num_labels):

    inputs = Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(20, activation="relu")(x)
    x = Dropout(0.1)(x)
    outputs = Dense(num_labels, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def build_model_pretrained(MAX_NB_WORDS, EMBEDDING_DIM, embedding_matrix, MAX_SEQUENCE_LENGTH, num_labels):
    from keras.layers import Embedding

    embedding_layer = Embedding(MAX_NB_WORDS + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    print("embedded_sequence", embedded_sequences.shape)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    print("Conv1", x.shape)
    x = MaxPooling1D(5)(x)
    # x = Conv1D(128, 5, activation='relu')(x)
    # print("Conv2", x.shape)
    # x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    print("Conv3", x.shape)
    x = MaxPooling1D(35)(x)  # global max pooling
    print("MaxPooling1D", x.shape)
    x = Flatten()(x)
    print("flatten", x.shape)
    x = Dense(128, activation='relu')(x)
    preds = Dense(num_labels, activation='softmax')(x)
    model = Model(sequence_input, preds)
    return model
