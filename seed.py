# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 13:01:11 2017

@author: shrut
"""

from collections import defaultdict
import nltk
import codecs

# Initialize paths for training data set
train_data = "./traindata.text"
train_labels = "./traindata.label"

#Stop-words
stop_words = nltk.corpus.stopwords.words('english')
more_stop_words = [",",".",":","@","#",";","&","-","(",")"]
stop_words.extend(more_stop_words)

#Read data and tokenize it
fpText = codecs.open(train_data,'r',encoding='utf8')
fpLabel = codecs.open(train_labels,'r',encoding='utf8')
content = fpText.read()
labels  = fpLabel.read()


word_count = defaultdict(float)
seed = {}

for (line,label) in zip(content.split("\n")[:10000],labels.split("\n")[:10000]):     
    for w in nltk.word_tokenize(line):
        if w not in stop_words: # Remove stop-words
            if(label not in seed):
                seed[label]=defaultdict(float)
            seed[label][w]+=1

result = []
i=0
for l,w in seed.items():
    result.append(sorted(w.items() , key=lambda t : t[1] , reverse=True))
    
###### From this, we now have the top words of each category and we can use them to associate features.
