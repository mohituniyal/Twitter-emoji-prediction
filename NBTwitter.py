# Twitter emoji prediction 
from collections import defaultdict
import nltk
import codecs

def fake_features(word):
    return {'fake': word}


# Initialize paths for training data set
train_data = "./traindata/traindata.text"
train_labels = "./traindata/traindata.labels"

#Stop-words
stop_words = nltk.corpus.stopwords.words('english')
more_stop_words = [",",".",":","@","#"]
stop_words.extend(more_stop_words)

#Read data and tokenize it
fpText = codecs.open(train_data,'r',encoding='utf8')
fpLabel = codecs.open(train_labels,'r',encoding='utf8')
content = fpText.read()
labels  = fpLabel.read()
featuresets = []
labelled_words = []
for (line,label) in zip(content.split("\n"),labels.split("\n")):
    for w in nltk.word_tokenize(line):
        if w not in stop_words: # Remove stop-words
            labelled_words.append((w,label))

featuresets = [(fake_features(word), label) for (word, label) in labelled_words]
classifier = nltk.NaiveBayesClassifier.train(featuresets)

#print classifier.classify(fake_features('great'))
