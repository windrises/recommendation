# coding=utf-8
from gensim.models import doc2vec
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
num = '1'
documents = doc2vec.TaggedLineDocument('document5.txt')
model = doc2vec.Doc2Vec(documents, size=500, window=1, min_count=500, workers=4)
model.save('./document' + num + '.bin')

