import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from models import build_transformer_model, build_model_pretrained
import matplotlib.pyplot as plt

TEXT_DATA_DIR = "/Users/guoqiong/data/20_newsgroup/"
MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 1000
GLOVE_DIR="/Users/guoqiong/data/glove.6B/glove.6B/"
EMBEDDING_DIM=50

texts = []  # list of text samples
labels_index = {} # dictionary mapping label name to numeric id
index_labels = {} # mapping label bame back
labels = []  # list of label ids
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    print(name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        index_labels[label_id] = name
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                if sys.version_info < (3,):
                    f = open(fpath)
                else:
                    f = open(fpath, encoding='latin-1')
                t = f.read()
                # print(len(t))
                # print((t[:100]))
                i = t.find('\n\n')  # skip header
                if 0 < i:
                    t = t[i:]
                texts.append(t)
                f.close()
                labels.append(label_id)
print('Found %s texts.' % len(texts))

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

print(sequences[:2])

word_index = tokenizer.word_index
index_word = tokenizer.index_word

print('Found %s unique tokens.' % len(word_index))
print('Found %s unique index.' % len(index_word))

print(tokenizer.word_index['hello'])
print(tokenizer.index_word[100])

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
# len(labels)=19997
VALIDATION_SPLIT=0.7
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

def get_glove_embeddings(GLOVE_DIR, EMBEDDING_DIM):
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.'+str(EMBEDDING_DIM)+'d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index

embeddings_index = get_glove_embeddings(GLOVE_DIR, EMBEDDING_DIM)
print('Found %s word vectors.' % len(embeddings_index))
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector



# 10 epochs, rmsprop, val acc =0.6226
model = build_transformer_model(maxlen=MAX_SEQUENCE_LENGTH,
                                vocab_size=MAX_NB_WORDS,
                                embed_dim=32,
                                num_heads=2,
                                ff_dim=32,
                                num_labels=len(labels_index))


print(model.summary())
# 10 epochs, rmsprop, val_acc =
model = build_model_pretrained(len(word_index), EMBEDDING_DIM,
                                embedding_matrix, MAX_SEQUENCE_LENGTH, len(labels_index))
print(model.summary())
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=10, batch_size=128)

history_dict = history.history
print(history_dict.keys())

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)
fig = plt.figure(figsize=(10, 6))
fig.tight_layout()

plt.subplot(2, 1, 1)
# "bo" is for "blue dot"
plt.plot(epochs, loss, 'r', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
# plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# input_text = "hello, how are you, this is Benjamin saying hi from the space station. I donot feel gravity here, and it is intresting"
# input_seq = tokenizer.texts_to_sequences([input_text])
# input_seq = pad_sequences(input_seq, maxlen=MAX_SEQUENCE_LENGTH)
# predictions = model.predict(input_seq)
# prediction_classes = np.argmax(predictions, axis=1)
#
# print(prediction_classes)
# print(index_labels[prediction_classes[0]])