#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:10:36 2017

@author: shruthi
"""
from collections import defaultdict
#import nltk
import codecs
import matplotlib.pyplot as plt

count = 9000
label_count = defaultdict(float)

train_data = "./traindata.text"
train_labels = "./traindata.label"
small_train_data = "./balanced_traindata.text"
small_train_label = "./balanced_trainlabel.label"

fptrain = codecs.open(train_data,'r',encoding='utf8')
fpLabel = open(train_labels,'r')

fp_small_data = codecs.open(small_train_data,'w',encoding='utf8')
fp_small_label = open(small_train_label,'w')

train = fptrain.read()
label = fpLabel.read()

for (line,label) in zip(train.split("\n"),label.split("\n")):
    if label_count[label] <= count:
        fp_small_data.write(line+"\n")
        fp_small_label.write(label+"\n")
        label_count[label]+=1

        
fp_small_data.close()
fp_small_label.close()



