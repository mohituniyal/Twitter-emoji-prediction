# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 15:43:32 2017

@author: shrut
"""

from collections import defaultdict
#import nltk
import codecs
import matplotlib.pyplot as plt

# Initialize paths for training data set
train_data = "./traindata.text"
train_labels = "./traindata.label"

fpLabel = codecs.open(train_labels,'r',encoding='utf8')
labels  = fpLabel.read()
count = defaultdict(float)
for label in zip(labels.split("\n")):
    if label==',':
        continue
    count[label]+=1
    
print count


x = []
y = []
X_LABEL = "rank"
Y_LABEL = "frequency"

# implement me! you should fill the x and y arrays. Add your code here
sorted_list = sorted(count.items(), key=lambda(k,v): v, reverse=True)
rank = 1
for n,v in sorted_list:
    x.append(rank)
    rank += 1
    y.append(v)

plt.scatter(x, y)
plt.xlabel(X_LABEL)
plt.ylabel(Y_LABEL)
