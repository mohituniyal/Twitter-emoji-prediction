# Twitter emoji prediction 
from collections import defaultdict
import nltk
import codecs
import string
from seeds import getSeedWords
from confusion_matrix import getAccuracy

# Initialize paths for training data set
train_data = "./traindata.text"
train_labels = "./traindata.label"
test_data = "./testdata.text"
test_labels = "./testdata.labels"

def fake_features(word):
    return {'fake': word}

#Stop-words
stop_words = nltk.corpus.stopwords.words('english')
more_stop_words = [",",".",":","@","#",";","&","-","(",")","user","...", "!", "'s","?","--","|","``","''","[","]","'","$",]
stop_words.extend(more_stop_words)
    
#Read data and tokenize it
fpText  = codecs.open(train_data,'r',encoding='utf8')
fpLabel = codecs.open(train_labels,'r',encoding='utf8')
content = fpText.read()
labels  = fpLabel.read()
fpText.close()
fpLabel.close()
featuresets = []
labelled_words = []
document = []
all_words = []
seed_words = getSeedWords(content,labels)

for (line,label) in zip(content.split("\n")[:10000],labels.split("\n")[:10000]):
    tweet_words = []
    for w in nltk.word_tokenize(line):
        w = w.lower()
        if w not in stop_words:
            # Remove stop-words
            if w not in string.punctuation:
                if w not in string.digits:
            #labelled_words.append((w,label))
                    tweet_words.append(w)    
                    all_words.append(w)
    document.append((tweet_words,label))


freq_words = nltk.FreqDist(w.lower() for w in all_words)
freq_words = sorted(freq_words.keys(), key=freq_words.get, reverse = True)
word_features = list(freq_words)[:5000]

def document_features(document): 
    document_words = set(document) 
    features = {}
    for word in word_features:
        #print word
        features['contains(%s)'%(word)] = (word in document_words)
    for index in xrange(len(seed_words)):
        for word in seed_words[index]:
            features['contains_%d'%(index)] = (word in document_words)
                    
    return features
    
#featuresets = [(fake_features(word), label) for (word, label) in labelled_words]
featuresets = [(document_features(d), c) for (d,c) in document]

classifier = nltk.NaiveBayesClassifier.train(featuresets)

#d,c = document[24]
#print d
#print classifier.classify(document_features(d))

fpText  = codecs.open(test_data,'r',encoding='utf8')
fpLabel = codecs.open(test_labels,'r',encoding='utf8')
content = fpText.read()
labels  = fpLabel.read()
fpText.close()
fpLabel.close()
document = []

for (line,label) in zip(content.split("\n")[:10],labels.split("\n")[:10]):
    tweet_words = []
    for w in nltk.word_tokenize(line):
        w = w.lower()
        if w not in stop_words: # Remove stop-words
             if w not in string.punctuation:
                if w not in string.digits:
                    tweet_words.append(w)
    document.append((tweet_words,label))

test_set = [(document_features(d), c) for (d,c) in document]
print "Accuracy: ",nltk.classify.accuracy(classifier, test_set)

classifier.show_most_informative_features(5)

errors = []
correct = []
guess = []
for (line, label) in zip(content.split("\n")[:10],labels.split("\n")[:10]):
    guess.append(classifier.classify(document_features(line)))
    
accuracy,precision, recall = getAccuracy(guess,labels.split("\n")[:10])
    

   
