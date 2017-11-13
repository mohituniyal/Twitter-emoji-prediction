# Twitter emoji prediction 
from collections import defaultdict
import nltk
import codecs

# Initialize paths for training data set
train_data = "./traindata/traindata.text"
train_labels = "./traindata/traindata.labels"

#Stop-words
stop_words = nltk.corpus.stopwords.words('english')
more_stop_words = [",",".",":","@"]
stop_words.extend(more_stop_words)

#Make bow as a dict of dicts
bow = {'0':defaultdict(float),
    '1':defaultdict(float),
    '2':defaultdict(float),
    '3':defaultdict(float),
    '4':defaultdict(float),
    '5':defaultdict(float),
    '6':defaultdict(float),
    '7':defaultdict(float),
    '8':defaultdict(float),
    '9':defaultdict(float),
    '10':defaultdict(float),
    '11':defaultdict(float),
    '12':defaultdict(float),
    '13':defaultdict(float),
    '14':defaultdict(float),
    '15':defaultdict(float),
    '16':defaultdict(float),
    '17':defaultdict(float),
    '18':defaultdict(float),
    '19':defaultdict(float)}

#Read data and tokenize it
fpText = codecs.open(train_data,'r',encoding='utf8')
fpLabel = codecs.open(train_labels,'r',encoding='utf8')
content = fpText.read()
labels  = fpLabel.read()
for (line,label) in zip(content.split("\n"),labels.split("\n")):
    for w in nltk.word_tokenize(line):
        if w not in stop_words:
            #print label,":",w
            bow[label][w] += 1
            
