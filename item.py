#coding=utf-8
#!/usr/bin/env python
import os
import json
import sys
import numpy as np
import re
reload(sys)
sys.setdefaultencoding('utf-8')

f = open('./item.txt')
out = open('./item_recommend.txt', 'w')
lines = f.readlines()
n = len(lines) - 1
m = int(lines[0].strip('\n'))
print n, m
tag_mtx = [[] for i in range(m)]
for i in range(1, n + 1):
    line = lines[i].strip('\n')
    line = line.split(' ')
    for x in line:
        x = x.split(',')
        tag_mtx[int(x[1])].append([i - 1, int(x[0])])
print 'start calculate'
sub_mtx = [{} for i in range(n)]
for tag in tag_mtx:
    for x in tag:
        for y in tag:
            if x == y:
                continue
            if y[0] not in sub_mtx[x[0]]:
                sub_mtx[x[0]][y[0]] = min(x[1], y[1])
            else:
                sub_mtx[x[0]][y[0]] += min(x[1], y[1])
print 'start recommend'
for i in range(n):
    sub_mtx[i] = sorted(sub_mtx[i].items(), key=lambda x: x[1], reverse=True)
    sub_mtx[i] = sub_mtx[i][:50]
    s = ''
    for x in sub_mtx[i]:
        s += str(x[0]) + ',' + str(x[1]) + ' '
    s = s[:-1]
    out.write(s + '\n')
out.close()
