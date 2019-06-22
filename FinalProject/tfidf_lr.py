import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression

from FinalProject import load_data

TRAIN_DATA_SHAPE = 19313
TEST_DATA_SHAPE = 8283
MAX_FEATURE = 2000
STOP_WORDS = "data/stop_words.txt"
OUTPUT_PATH = "data/LinearRegression_output.csv"


if __name__ == '__main__':
    text = load_data.text
    score = load_data.text_score

    test_text = load_data.test_text
    test_score = load_data.test_score

    stop_words = list()
    with open(STOP_WORDS, 'r', encoding="utf8") as f:
        for line in f.readlines():
            linestr = line.strip()
            stop_words.append(linestr)

    input = []
    for txt in text:
        for word in jieba.cut(txt, cut_all=True):
            if word != "":
                input.append(word)

    test_input = []
    for txt in test_text:
        for word in jieba.cut(txt, cut_all=True):
            if word != "":
                test_input.append(word)
    # for word in input:
    #     print(word)
    # print(input)

    tfidf_vec = TfidfVectorizer(stop_words=stop_words, max_features=MAX_FEATURE)
    tfidf = tfidf_vec.fit_transform(input)

    x_train = tfidf[0: TRAIN_DATA_SHAPE]
    # print(x_train)
    linear_regression = LinearRegression()
    linear_regression.fit(x_train, score)

    tfidf = tfidf_vec.fit_transform(test_input)
    x_test = tfidf[0: TEST_DATA_SHAPE]
    # print(x_test)

    predictions = linear_regression.predict(x_test)
    ids = load_data.ids

    # for prediction in predictions:


    df = pd.DataFrame(predictions, columns=["prediction"])
    df.to_csv(OUTPUT_PATH)

    res = 4.164657031
