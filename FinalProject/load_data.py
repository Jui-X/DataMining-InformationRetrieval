import json
import numpy as np

import warnings

from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

warnings.filterwarnings(action="ignore", category=UserWarning, module="gensim")

DATA_PATH = 'data/user_seq.json'
TRAIN_DATA = "data/train.csv"
TEST_DATA = "data/test.csv"
MODEL_PATH = "model.h5"
MAX_WORDS_NUM = 20000
MAX_LEN = 20


# if __name__ == '__main__':
user_seq = json.load(open(DATA_PATH, 'r'))

scores = []
with open(TRAIN_DATA, "r") as f:
    train_data = f.readlines()
    scores = []
    for raw_data in train_data:
        raw = raw_data.split(',')
        scores.append({raw[0]: raw[1].strip("\n")})



user_tweet = []
text = []
text_score = []
time = []
user = []
for sequence in user_seq:
    for seq in sequence:
        for score in scores:
            if seq["id_str"] in score.keys() and seq["text"] != "":
                # user_tweet.append({"post_content": seq["text"], "post_time": seq["time"],
                user.append(str(seq["user"]))
                text.append(str(seq["text"]).strip("\n"))
                text_score.append(score[seq["id_str"]])
                time.append(str(seq["time"]))
                break

tokenizer = Tokenizer(num_words=MAX_WORDS_NUM)
tokenizer.fit_on_texts(text)
# print(tokenizer.word_index)
x_seq = tokenizer.texts_to_sequences(text)
txt_input = pad_sequences(x_seq, maxlen=MAX_LEN)
# print(txt_input)
user_input = tokenizer.texts_to_sequences(user)
user_input = pad_sequences(user_input, maxlen=MAX_LEN)
# print(user_input)
time_input = tokenizer.texts_to_sequences(time)
time_input = pad_sequences(time_input, maxlen=MAX_LEN)
# print(time_input)
score_input = np.array(text_score)
# print(y_train.shape)

with open(TEST_DATA, "r") as ft:
    test_data = ft.readlines()
    test_scores = []
    ids = []
    for raw_data in test_data:
        raw = raw_data.split(',')
        test_scores.append({raw[0]: raw[1].strip("\n")})
        id = {"id": raw[0]}
        ids.append(id)

test_user = []
test_text = []
test_score = []
test_time = []
for user in user_seq:
    for seq in user:
        for score in test_scores:
            if seq["id_str"] in score.keys() and seq["text"] != "":
                # user_tweet.append({"post_content": seq["text"], "post_time": seq["time"],
                test_user.append(str(seq["user"]))
                test_text.append(str(seq["text"]).strip("\n"))
                test_score.append(score[seq["id_str"]])
                test_time.append(str(seq["time"]))
                break

tokenizer = Tokenizer(num_words=MAX_WORDS_NUM)
tokenizer.fit_on_texts(test_text)
# print(tokenizer.word_index)
test_seq = tokenizer.texts_to_sequences(test_text)
test_txt = pad_sequences(test_seq, maxlen=MAX_LEN)
# print(test_txt)
test_user = tokenizer.texts_to_sequences(test_user)
test_user = pad_sequences(test_user, maxlen=MAX_LEN)
# print(test_user)
test_time = tokenizer.texts_to_sequences(test_time)
test_time = pad_sequences(test_time, maxlen=MAX_LEN)
# print(test_time)
test_score = np.array(test_score)
# print(y_test)


# def word2vec():
    # model = Word2Vec(sentences=texts, size=EMBEDDING_DIM, window=WINDOW, min_count=MIN_COUNT, sg=SKIP_GRAM)
    #
    # print(model)
    #
    # model.wv.most_similar(positive="you", topn=5)
    #
    # model.save("w2v.model")