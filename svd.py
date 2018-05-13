import numpy as np
from numpy import linalg
import pandas as pd
import seaborn

data = pd.read_csv('./bgmdata/ratings.txt', sep=' ', engine='python', names=['uid', 'sid', 'rating']) # read data from file
n_user = data.uid.unique().shape[0] # number of users
n_sub = data.sid.unique().shape[0] #  number of subjects
print n_user, n_sub

X = np.zeros((n_user, n_sub), dtype='float64') # rating matrix
f = open('ratings.txt')
for line in input:
    line = line.strip('\r\n').split(' ')
    uid = int(line[0])
    sid = int(line[1])
    rating = float(line[2])
    X[uid, sid] = rating
sub_avg = [] # subject average rating
for i in range(n_sub):
    sub_avg.append(np.sum(X[:, i]) / np.count_nonzero(X[:, i]))
for i in range(n_user):
    for j in range(n_sub):
        if X[i, j] == 0:
            X[i, j] = sub_avg[j]
for i in range(n_user):
    X[i] -= np.sum(X[i]) / np.count_nonzero(X[i]) # Standardized

U, S, V = linalg.svd(X)

recommend = [[] for i in range(n_sub)]
for i in range(n_sub):
    for j in range(n_sub):
        if i == j:
            continue
        vi = V[:, i]
        vj = V[:, j]
        recommend[i].append([j, np.dot(vi, vj) / (linalg.norm(vi) * linalg.norm(vj))]) # cosine similarity

out = open('./save/svd_recommend.txt', 'w')
for i in range(n_sub):
    recommend[i] = sorted(recommend[i], key=lambda x: x[1], reverse=True)
    s = ''
    topn = min(20, len(recommend[i])) # top 20
    for j in range(topn):
        s += str(recommend[i][j][0]) + ' ' + str(recommend[i][j][1]) + ','
    s = s[:-1]
    out.write(s + '\n')
out.close()
