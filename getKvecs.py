#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:32:22 2017

@author: mohituniyal
"""
# Twitter emoji prediction 

#import nltk
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import numpy as np
from gensim.models import word2vec
import unicodedata
import string


stop_words = stopwords.words('english')
more_stop_words = [",",".",":","@","#",";","&","-","(",")","user","...", "!", "'s","?","--","|","``","''","[","]","'","$",]
all_letters = string.ascii_letters + " .,;'"

def unicodeToAscii(s):
    return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn' and c in all_letters
    )
    
stop_words_ascii = []
for s in stop_words:
    stop_words_ascii.append(unicodeToAscii(s))
    
stop_words_ascii.extend(more_stop_words)

# Defining a set of ascii characters that we are going to keep
printable = set(string.printable)

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
        stops = set(stop_words_ascii)
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)

# Define a function to split a review into parsed sentences
def review_to_sentences( review, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    raw_sentences = review.split("\n")
    sentences = []
    counter = 0
    for raw_sentence in raw_sentences:
        counter += 1
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, remove_stopwords ))
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


# Returns vectors for words
# Params:   k : 
#           num_feat : number of features in each word-vector
#           train_file : file with training data
#           test_file : file with training data
#           val_file : file with training data
def getKwordVecs(k=1,num_feat=512,train_file="balanced_traindata.text",test_file="",val_file=""):
    
    convert2vecs = []
    sentences = []  # Initialize an empty list of sentences
    
    # List the files we need to read
    if train_file != "":
        convert2vecs.append(train_file)
    if test_file != "":
        convert2vecs.append(test_file)        
    if val_file != "":
        convert2vecs.append(val_file)
    
    for i in convert2vecs:
        #Open Training files, get content and close files
        f_d = open(i,'r')
        t_data   = filter(lambda x: x in printable, f_d.read())
        f_d.close()
        
        sentences.extend(review_to_sentences(t_data))
    
    ##### Training the word2vec model
    
    # Import the built-in logging module and configure it so that Word2Vec 
    # creates nice output messages
    if sentences == []:
        return "Error"
        
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)
    
    # Set values for various parameters
    num_features = num_feat    # Word vector dimensionality                      
    min_word_count = 40   # Minimum word count                        
    num_workers = 10      # Number of threads to run in parallel
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
    
    trainDataVecs = getAvgFeatureVecs( sentences[:k], w2v_model, num_features )
    return (w2v_model, trainDataVecs)
