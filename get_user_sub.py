import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Embedding, Reshape, Merge
from keras.layers import Conv2D, MaxPooling2D
from keras.models import model_from_json
import numpy as np

def get_recommendation(i, embeddings, top_n, out):
    norms = np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings, axis=1)
    cosine = np.dot(embeddings, embeddings[i]) / norms
    index = np.argsort(-cosine)
    s = ''
    for x in range(top_n):
        if index[x] != i:
            s += str(index[x]) + ',' + str(cosine[index[x]]) + ' '
    s = s[:-1]
    out.write(s + '\n')

model = keras.models.load_model('./save/train/model.h5')
weights = model.get_weights()
user_embeddings = weights[0]
sub_embeddings = weights[1]
print user_embeddings.shape, sub_embeddings.shape
print user_embeddings[0].shape
out = open('./user_recommend.txt', 'w')
for i in range(28574):
    if i % 100 == 0:
        print i
    get_recommendation(i, user_embeddings, 100, out)
out.close()
out = open('./sub_recommend.txt', 'w')
for i in range(8518):
    if i % 100 == 0:
        print i
    get_recommendation(i, sub_embeddings, 50, out)
out.close()
