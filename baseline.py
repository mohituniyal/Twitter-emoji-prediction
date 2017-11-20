# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 15:42:08 2017

@author: shrut
"""

# Twitter emoji prediction 
import codecs

# Initialize paths for training data set
train_data = "./traindata.text"
train_labels = "./traindata.label"
test_data = "./testdata.text"
test_labels = "./testdata.labels"
val_labels = "./valdata.labels"
#Read data and tokenize it
fpLabel = codecs.open(train_labels,'r',encoding='utf8')
labels  = fpLabel.read()
fpLabel.close()

predict = '0'
correct = 0.0
total = 0.0

for label in labels.split("\n"):
    if predict == label:
        correct +=1.0
    total+=1.0

#Read data and tokenize it
fpLabel = codecs.open(val_labels,'r',encoding='utf8')
labels  = fpLabel.read()
fpLabel.close()

for label in labels.split("\n"):
    if predict == label:
        correct +=1
    total+=1.0
    
baseline = correct/total

print "Baseline Accuracy:",baseline



