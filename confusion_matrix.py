# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:59:02 2017

@author: shrut
"""

from collections import defaultdict

def getAccuracy(predictions,gold_labels):
    all_labels = set(gold_labels)
    
    true_positive = defaultdict(float)
    true_negative = defaultdict(float)
    false_positive = defaultdict(float)
    false_negative = defaultdict(float)
    
    for p,g in zip(predictions,gold_labels):
        if p==g:
            true_positive[p]+=1
            others = filter(lambda x: x != p, all_labels)
            for o in others:
                true_negative[o]+=1 
        else:
            false_positive[p]+=1
            false_negative[g]+=1
            others = filter(lambda x: x not in (p,g), all_labels)
            for o in others:
                true_negative[o]+=1
        
    accuracy = defaultdict(float)
    precision = defaultdict(float)
    recall = defaultdict(float)
       
    for label in all_labels:
        tptn = (true_positive[label]+true_negative[label])
        accuracy[label] = tptn/(tptn+false_negative[label]+false_positive[label]) 
        if(true_positive[label]!=0):
            precision[label] = true_positive[label]/(true_positive[label]+false_positive[label])
            recall[label] = true_positive[label]/(true_positive[label]+false_negative[label])
        else:
            precision[label],recall[label]
        
    return (accuracy,precision,recall)
            
predictions = [0,1,2,0,1,2]
gold_labels = [0,1,2,4,5,6]

