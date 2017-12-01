#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:32:22 2017

@author: mohituniyal
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 11:00:21 2017

@author: mohituniyal
"""

# Twitter emoji prediction 
#from collections import defaultdict
import nltk
import codecs
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

import nltk.data  
from gensim.models import word2vec


# Define a function to split a review into into a list of words
def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()
    #  
    # 2. Remove non-letters
    review = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)

# Define a function to split a review into parsed sentences
def review_to_sentences( review, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    #raw_sentences1 = tokenizer.tokenize(review)
    #print "old_len:",len(raw_sentences1)
    raw_sentences = review.split("\n")
    #print "new_len:",len(raw_sentences)
    # 2. Break sentences into separate 
    #raw_sentences = raw_sentences.split("\n")
    #
    # 3. Loop over each sentence
    sentences = []
    counter = 0
    for raw_sentence in raw_sentences:
        counter += 1
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, \
              remove_stopwords ))
        if (counter%10000 == 0):
            print "Seen %d tweets"%counter
        
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    # 
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.wv.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    # 
    # Divide the result by the number of words to get the average
    if nwords == 0:
        return featureVec 
    else:
        return np.divide(featureVec,nwords)


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0
    # 
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    # 
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%10000 == 0:
           print "Review %d of %d" % (counter, len(reviews))
       # 
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
       #
       # Increment the counter
       counter = counter + 1
    return reviewFeatureVecs


def getKwordVecs(k=1,file_name="balanced_traindata.text"):
    
    # Initialize paths for training data set
    train_data = file_name
    #train_labels = "traindata.labels"
    
    #Open Training files, get content and close files
    ftr_d = codecs.open(train_data,'r',encoding='utf-8')
    #ftr_l = open(train_labels,'r')#,encoding='utf-8')
    tr_data   = ftr_d.read()
    #tr_labels = ftr_l.read()
    ftr_d.close()
    #ftr_l.close()
    
    print "All set with the files now"
    
    # Load the punkt tokenizer
    #tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    
    sentences = []  # Initialize an empty list of sentences
    print "Parsing sentences from training set"
    sentences = review_to_sentences(tr_data)
    
    #print "len of sentences:",len(sentences)
    
    #print sentences[0]
    
    
    ##### Training the word2vec model
    
    # Import the built-in logging module and configure it so that Word2Vec 
    # creates nice output messages
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)
    
    # Set values for various parameters
    num_features = 500    # Word vector dimensionality                      
    min_word_count = 40   # Minimum word count                        
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size                                                                                    
    downsampling = 1e-3   # Downsample setting for frequent words
    
    # Initialize and train the model (this will take some time)
    
    print "Training model..."
    w2v_model = word2vec.Word2Vec(sentences, workers=num_workers, \
                size=num_features, min_count = min_word_count, \
                window = context, sample = downsampling)
    
    # If you don't plan to train the model any further, calling 
    # init_sims will make the model much more memory-efficient.
    w2v_model.init_sims(replace=True)
    
    # It can be helpful to create a meaningful model name and 
    # save the model for later use. You can load it later using Word2Vec.load()
    #model_name = "300features_40minwords_10context"
    #model.save(model_name)
    
    # ****************************************************************
    # Calculate average feature vectors for training and testing sets,
    # using the functions we defined above. Notice that we now use stop word
    # removal.
    #print "Creating average feature vecs for training set"
    #clean_train_reviews = review_to_sentences( tr_data, tokenizer, \
            #remove_stopwords=True )
    
    trainDataVecs = getAvgFeatureVecs( sentences[:k], w2v_model, num_features )
    return (w2v_model, trainDataVecs)

