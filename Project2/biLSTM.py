from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb


MAX_FEATURES = 20000
# cut texts after this number of words
# (among top max_features most common words)
maxlen = 100
BATCH_SIZE = 32


def load_data():
    print('Loading data...')
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    with open("data/training_set.ss", "r", encoding="utf8") as f:
        lines = f.readlines()
        for line in lines:
            text = line.split("\t")
            x_train.append(text[3].strip("\n"))
            y_train.append(text[2])
    with open("data/validation_set.ss", "r", encoding="utf8") as f:
        lines = f.readlines()
        for line in lines:
            text = line.split("\t")
            x_test.append(text[3].strip("\n"))
            y_test.append(text[2])
    return (x_train, y_train), (x_test, y_test)


def biLSTM():
    (x_train, y_train), (x_test, y_test) = load_data()
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    model = Sequential()
    model.add(Embedding(MAX_FEATURES, 128, input_length=maxlen))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=4,
              validation_data=[x_test, y_test])


if __name__ == "__main__":
    biLSTM()