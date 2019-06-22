import csv

import pandas as pd
import json
import jieba
import codecs
from gensim import corpora
from gensim.models import LdaModel
from gensim.corpora import  Dictionary
import os
import sys


stop_words_path = 'data/stop_words.txt'
data_path = 'data/user_seq.json'
train_path = 'data/train.csv'
test_path = 'data/test.csv'
topic_num = 150
passes = 1
user_rate = 0
topic_rate = 1-user_rate
# topic_num   passes   user_rate     mse
# 50          1        0.8         1.2448
# 50          1        0.9         1.2268
# 50          1        1           1.2250
# 50          100      0.8         1.2433
# 50          20       0.8         1.2442
# 30          20       0.8         1.2448
# 80          20       0.8         1.2448
# 60          20       0.8         1.2444
# 60          20       0           1.8296
# 60          200      0           1.8302
# 10          1        1           1.2250
# 10          1        0           1.84
# 10          20        0          1.8348



# 预处理过程
def data_preprocess(user_seq, train, test):
    # 向第一个数据集添加热度score属性
    for i in range(0, len(user_seq)):
        for j in range(0, len(user_seq[i])):
            if pd.isna(train[train['userId'] == int(user_seq[i][j]['id_str'])]).empty == False:
                user_seq[i][j]['score'] = train[train['userId'] == int(user_seq[i][j]['id_str'])]['score'].iloc[0]
            else:
                user_seq[i][j]['score'] = None
    # 去除text中的停用词
            user_seq[i][j]['text'] = seg_sentence(user_seq[i][j]['text']).strip()
    return user_seq


def get_csv_file(path):
    data_set = pd.read_csv(path, header=None)
    data_set.columns = ['userId', 'score']
    return data_set


def get_json_file(path):
    data_set = json.load(open(path, 'r'))
    return data_set


# 读取停用词文件
def stop_words_list(path):
    stopwords = [line.strip() for line in open(path,'r', encoding="utf8").readlines()]
    return stopwords


# 分句函数，去除句子中的停用词
def seg_sentence(sentence):
    sentence_seged = jieba.cut(sentence.strip())
    stop_words = stop_words_list(stop_words_path)
    outstr = ''
    for word in sentence_seged:
        word = word.lower()
        if word not in stop_words:
            if word != '\t':
                outstr += word
                outstr += " "
    outstr = ' '.join(outstr.split())
    return outstr


def max_topic(list):
    max_index = 0
    max_rate = 0
    i = 0
    for tuple in list:
        if tuple[1] > max_rate:
            max_rate = tuple[1]
            max_index = i
        i += 1
    topic = list[max_index][0]
    return topic


if __name__ == '__main__':
    topic_num = 12
    # passes = int(sys.argv[2])
    # user_rate = int(sys.argv[3])
    topic_rate = 1 - user_rate
    # 取数据过程
    train = get_csv_file(train_path)
    test = get_csv_file(test_path)
    user_seq = get_json_file(data_path)
    # # 数据预处理
    user_seq = data_preprocess(user_seq, train, test)
    # 得到无停用词文件
    fp = open('data/noStopWordData', 'w', encoding='utf-8')
    for items in user_seq:
        for item in items:
            item = json.dumps(item)
            fp.write(item+'\n')
    user_seq = open('data/noStopWordData', 'r', encoding='utf-8').readlines()
    # 构建字典，用于Lda模型：
    te = []
    for line in user_seq:
        line = json.loads(line)
        te.append([w for w in line['text'].split()])
    dictionary = corpora.Dictionary(te)
    corpus = [dictionary.doc2bow(text) for text in te]
    # num_topics 必须。生成的主题数量
    # id2word 必须。把id 都映射为字符串
    # passes 可选。模型遍历语料库的次数，次数越多越精确，但是遍历太多次花费时间长
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=topic_num, passes=passes)
    # 添加主题到user_seq
    fp = open('data/topic_result', 'w', encoding='utf-8')
    # 将有主题的数据存文件
    # topic_result = []
    for item in user_seq:
        item = json.loads(item)
        item['topic'] = max_topic(lda[dictionary.doc2bow(item["text"].split())])
        fp.write(json.dumps(item)+'\n')
    fp.close()
    topic_result = []
    fp = open('data/topic_result', 'r', encoding='utf-8')
    user_Ids = set()
    topic_Ids = set()
    user_Ids_average_score = {}
    topic_Ids_average_score = {}
    # 遍历数据，初始化平均分字典的值为0

    for item in fp.readlines():
        item = json.loads(item)
        user_Ids_average_score[item['user']] = 0
        user_Ids_average_score[item['user']+"Count"] = 0
        topic_Ids_average_score[item['topic']] = 0
        topic_Ids_average_score[str(item['topic'])+"count"] = 0
        user_Ids.add(item['user'])
        topic_Ids.add(item['topic'])
        topic_result.append(item)
    user_Ids = list([user_Ids][0])
    topic_Ids = list([topic_Ids][0])
    # 再次遍历数据，计算对应的score总和及个数
    for item in topic_result:
        if item['score'] is not None:
            user_Ids_average_score[item['user']] += item['score']
            user_Ids_average_score[item['user']+"Count"] += 1
            topic_Ids_average_score[item['topic']] += item['score']
            topic_Ids_average_score[str(item['topic'])+'count'] += 1
    for user_Id in user_Ids:
        user_Ids_average_score[user_Id] /= user_Ids_average_score[user_Id+"Count"]
    for topic_Id in topic_Ids:
        topic_Ids_average_score[topic_Id] /= topic_Ids_average_score[str(topic_Id)+'count']
    fp.close()
    test = get_csv_file(test_path)
    mse = 0
    i = 0
    predictions = []
    for item in test['userId']:
        for data in topic_result:
            if item == int(data['id_str']):
                predict_score = user_Ids_average_score[data['user']]*user_rate + topic_Ids_average_score[data['topic']]*topic_rate
                predictions.append(str(predict_score))
                mse += pow((predict_score - test['score'][i]), 2)
                i += 1

    print(predictions)
    df = pd.DataFrame(predictions, columns=["prediction"])
    df.to_csv("data/LDA_output.csv")

    mse = mse / 8283
    mse = pow(mse, 0.5)
    print(mse)

    os.remove("data/topic_result")
