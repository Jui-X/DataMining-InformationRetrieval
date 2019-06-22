import jieba
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

from FinalProject import load_data

STOP_WORDS = "data/stop_words.txt"

if __name__ == "__main__":
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

    tfidf_vec = TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 2), norm="l2")
    tfidf = tfidf_vec.fit_transform(input)
    lda = LatentDirichletAllocation(n_topics=12, max_iter=50, learning_method="batch")
    lda.fit(tfidf)

    test_input = []
    for txt in test_text:
        for word in jieba.cut(txt, cut_all=True):
            if word != "":
                test_input.append(word)

    prediction = lda.transform(test_input)
    print(prediction)