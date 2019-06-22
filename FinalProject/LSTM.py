import json

import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

DATA_PATH = "data/user_seq.json"
TRAIN_DATA = "data/train.csv"
TEST_DATA = "data/test.csv"
MODEL_PATH = "LSTM_model.h5"
OUTPUT_PATH = "LSTM_output.csv"
MAX_WORDS_NUM = 20000
BATCH_SIZE = 16
DROPOUT_VALUE = 0.15
MAX_LEN = 20


def load_data():
    user_seq = json.load(open(DATA_PATH, 'r'))

    scores = []
    with open(TRAIN_DATA, "r") as f:
        train_data = f.readlines()
        scores = []
        for raw_data in train_data:
            raw = raw_data.split(',')
            scores.append({raw[0]: raw[1].strip("\n")})
    # for score in scores:
    #     print(score)

    user_tweet = []
    text = []
    text_score = []
    for user in user_seq:
        for seq in user:
            for score in scores:
                if seq["id_str"] in score.keys() and seq["text"] != "":
                    # user_tweet.append({"post_content": seq["text"], "post_time": seq["time"],
                    # "score": score[seq["id_str"]]})
                    text.append(str(seq["text"]).strip("\n"))
                    text_score.append(score[seq["id_str"]])
                    break
    # for text in texts:
    #     print(text)
    # print(text_score)

    tokenizer = Tokenizer(num_words=MAX_WORDS_NUM)
    tokenizer.fit_on_texts(text)
    # print(tokenizer.word_index)
    x_seq = tokenizer.texts_to_sequences(text)
    x_train = pad_sequences(x_seq, maxlen=MAX_LEN)
    # print(x_train)
    y_train = np.array(text_score)
    # print(y_train)

    with open(TEST_DATA, "r") as ft:
        test_data = ft.readlines()
        test_scores = []
        for raw_data in test_data:
            raw = raw_data.split(',')
            test_scores.append({raw[0]: raw[1].strip("\n")})
    # for score in test_scores:
    #     print(score)

    test_text = []
    test_score = []
    for user in user_seq:
        for seq in user:
            for score in test_scores:
                if seq["id_str"] in score.keys() and seq["text"] != "":
                    # user_tweet.append({"post_content": seq["text"], "post_time": seq["time"],
                    # "score": score[seq["id_str"]]})
                    test_text.append(str(seq["text"]).strip("\n"))
                    test_score.append(score[seq["id_str"]])
                    break
    # for text in test_text:
    #     print(text)
    # print(test_score)

    tokenizer = Tokenizer(num_words=MAX_WORDS_NUM)
    tokenizer.fit_on_texts(test_text)
    # print(tokenizer.word_index)
    test_seq = tokenizer.texts_to_sequences(test_text)
    x_test = pad_sequences(test_seq, maxlen=MAX_LEN)
    # print(x_test)
    y_test = np.array(test_score)
    # print(y_test)

    return (x_train, y_train), (x_test, y_test)


def get_model():
    (x_train, y_train), (x_test, y_test) = load_data()

    model = Sequential()
    model.add(Embedding(input_dim=MAX_WORDS_NUM, output_dim=256))
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation="relu"))
    model.add(Dropout(DROPOUT_VALUE))
    model.add(Dense(1, activation="relu"))

    model.compile(optimizer="sgd", loss="mse")

    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=4)

    model.save(MODEL_PATH)

    predictions = model.predict(x_test, batch_size=BATCH_SIZE)

    result = []
    for predict in predictions:
        res = {}

    df = pd.DataFrame(predictions, columns=["prediction"])
    df.to_csv(OUTPUT_PATH)

    score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)

    print(score)


# def my_activation(x):
#     return K.sigmoid(x) * 12


if __name__ == "__main__":
    get_model()

