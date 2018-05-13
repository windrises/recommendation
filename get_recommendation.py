import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Embedding, Reshape, Merge
from keras.layers import Conv2D, MaxPooling2D
from keras.models import model_from_json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

num = 'train'
model = keras.models.load_model('./save/' + num + '/model.h5')
out = open('./recommendation.txt', 'w')

n_user = 28574
n_sub = 8518
uid = np.zeros((n_user * n_sub,))
sid = np.zeros((n_user * n_sub,))
cnt = 0
for i in range(n_user):
    for j in range(n_sub):
        uid[cnt] = i
        sid[cnt] = j
        cnt += 1
result = model.predict([uid, sid])

cnt = 0
for i in range(n_user):
    all = []
    for j in range(n_sub):
        all.append([j, result[cnt][0]])
        cnt += 1
    all = sorted(all, key=lambda x: x[1], reverse=True)
    s = ''
    for j in range(n_sub):
        s += ' ' + str(all[j][0]) + ',' + str(all[j][1])
    out.write(s + '\n')
    if i % 100 == 0:
        print i
