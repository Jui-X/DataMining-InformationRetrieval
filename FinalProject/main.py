import json

import jieba
import numpy as np
import pandas as pd

from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DATA_PATH = 'data/user_seq.json'
TRAIN_DATA = "data/train.csv"
TEST_DATA = "data/test.csv"
MODEL_PATH = "model.h5"
STOP_WORDS = "data/stop_words.txt"
DATA_FEATURE = "data/feature.csv"
MAX_WORDS_NUM = 20000
MAX_LEN = 20
TRAIN_DATA_SHAPE = 19313
TEST_DATA_SHAPE = 8283

if __name__ == '__main__':
    user_seq = json.load(open(DATA_PATH, 'r'))

    scores = []
    with open(TRAIN_DATA, "r") as f:
        train_data = f.readlines()

        for raw_data in train_data:
            score = {}
            raw = raw_data.split(',')
            score["id"] = raw[0]
            score["score"] = raw[1].strip("\n")
            scores.append(score)

    # with open(TRAIN_DATA, "r") as f:
    #     train_data = f.readlines()
    #     for raw_data in train_data:
    #         raw = raw_data.split(',')
    #         score = {"id": raw[0], "score": raw[1].strip("\n")}
    #         scores.append(score)

    df = pd.DataFrame(scores, columns=["id", "score"])
    df.to_csv(DATA_FEATURE, index=False)

    # user_tweet = []
    # text = []
    # text_score = []
    # time = []
    # for user in user_seq:
    #     for seq in user:
    #         for score in scores:
    #             if seq["id_str"] in score.keys() and seq["text"] != "":
    #                 # user_tweet.append({"post_content": seq["text"], "post_time": seq["time"],
    #                 # "score": score[seq["id_str"]]})
    #                 text.append(str(seq["text"]).strip("\n"))
    #                 text_score.append(score[seq["id_str"]])
    #                 time.append(str(seq["time"]))
    #                 break
    #
    # with open(TEST_DATA, "r") as ft:
    #     test_data = ft.readlines()
    #     test_scores = []
    #     for raw_data in test_data:
    #         raw = raw_data.split(',')
    #         test_scores.append({raw[0]: raw[1].strip("\n")})
    #
    # test_text = []
    # test_score = []
    # test_time = []
    # for user in user_seq:
    #     for seq in user:
    #         for score in test_scores:
    #             if seq["id_str"] in score.keys() and seq["text"] != "":
    #                 # user_tweet.append({"post_content": seq["text"], "post_time": seq["time"],
    #                 # "score": score[seq["id_str"]]})
    #                 test_text.append(str(seq["text"]).strip("\n"))
    #                 test_score.append(score[seq["id_str"]])
    #                 test_time.append(str(seq["time"]))
    #                 break

    # stop_words = list()
    # with open(STOP_WORDS, 'r', encoding="utf8") as f:
    #     for line in f.readlines():
    #         linestr = line.strip()
    #         stop_words.append(linestr)
    #
    # tfidf_vec = TfidfVectorizer(stop_words=stop_words)
    #
    # input = []
    # for txt in text:
    #     for word in jieba.cut(txt, cut_all=True):
    #         if word != "":
    #             input.append(word)
    #
    # test_input = []
    # for txt in test_text:
    #     for word in jieba.cut(txt, cut_all=True):
    #         if word != "":
    #             test_input.append(word)
    #
    # x_train = tfidf_vec.fit_transform(input)[0: TRAIN_DATA_SHAPE]
    # # print(tfidf)
    # y_train = text_score
    #
    # logistic_regression = LogisticRegression()
    # logistic_regression.fit(x_train, y_train)
    #
    # x_test = tfidf_vec.fit_transform(test_input)[0: TEST_DATA_SHAPE]
    #
    # logistic_regression.predict_proba(x_test)

    # print(text)
    # print(text_score)
    # print(time)
    # tokenizer = Tokenizer(num_words=MAX_WORDS_NUM)
    # tokenizer.fit_on_texts(text)
    # # print(tokenizer.word_index)
    # x_seq = tokenizer.texts_to_sequences(text)
    # x_train = pad_sequences(x_seq, maxlen=MAX_LEN)
    # # print(x_train.shape)
    # y_train = np.array(text_score)
    # # print(y_train.shape)
    # z_train = np.array(time)
    # # print(z_train)

    # tokenizer = Tokenizer(num_words=MAX_WORDS_NUM)
    # tokenizer.fit_on_texts(test_text)
    # # print(tokenizer.word_index)
    # test_seq = tokenizer.texts_to_sequences(test_text)
    # x_test = pad_sequences(test_seq, maxlen=MAX_LEN)
    # # print(x_test.shape)
    # y_test = np.array(test_score)
    # # print(y_test)
