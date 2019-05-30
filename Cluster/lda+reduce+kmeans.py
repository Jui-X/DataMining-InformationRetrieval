import json

from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from .main_mine import calculate_nmi

randomstate = 666


train_tokens_file = 'train_tokens.json'
train_topics_file = 'train_topics.json'
test_tokens_file = 'test_tokens.json'
output_file = '10152160145.json'


k_docid = 'docid'
k_topic = 'topic'
k_cluster = 'cluster'


def load_array(file):
    with open(file, 'r') as fp:
        array = [json.loads(line.strip()) for line in fp.readlines()]
    return array


train_tokens_list = load_array(train_tokens_file)
train_topics_list = load_array(train_topics_file)
test_tokens_list = load_array(test_tokens_file)

y = [item['topic'] for item in train_topics_list]

ids = [item[k_docid] for item in train_tokens_list]


word_vec = []
word_vec2 = []

for item in train_tokens_list:
	lst = [str(i) for i in item['tokenids']]
	word_vec.append(' '.join(lst))

for item in test_tokens_list:
	lst = [str(i) for i in item['tokenids']]
	word_vec2.append(' '.join(lst))


print('count vec...')
countVec = CountVectorizer()
cntTF = countVec.fit_transform(word_vec)
print('cntTF:')
print(cntTF)


print('lda...')
lda = LatentDirichletAllocation(n_topics=200, \
	learning_offset=50., random_state=randomstate)
docres = lda.fit_transform(cntTF)
# docres = []

cls = 65
# y2 = [line.argmax() for line in docres]
y2 = KMeans(n_clusters=cls, random_state=randomstate, \
	n_init=65).fit_predict(docres)


bat = [0 for i in range(65)]
for i in y:
	bat[i] += 1

bat2 = [0 for i in range(65)]
for i in y2:
	bat2[i] += 1

bat.sort()
bat2.sort()

for i in range(65):
	print(bat[i], ' and ', bat2[i])


cluster_list = [{k_docid: ids[i], k_cluster : y2[i]} for i in range(len(ids))]

calculate_nmi(train_topics_list, cluster_list)
