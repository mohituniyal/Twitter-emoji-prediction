#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import getKvecs
import random


N, D_in, H, D_out = "full", 300, 128, 20
dtype = torch.FloatTensor

###############################################

train_labels = "balanced_trainlabel.label"
#train_labels = traindata.label"
fpLabel = open(train_labels,'r')
fpdatastore = fpLabel.read()
fpLabel.close()
fpdata = []
for i in fpdatastore.strip().split("\n"):
    fpdata.append(i)
if N == "full":
    N = len(fpdata)
else:
    fpdata = fpdata[:N]

label_arr_tr = np.zeros((N,D_out),dtype=float)
#Making one-hot vectors for our data
for i in range(N):
    label_arr_tr[i][int(fpdata[i])] = 1

train_data_path = "balanced_traindata.text"
fptrain = open(train_data_path,'r')
train_data = fptrain.read()
fptrain.close()
train_data_list = []

for i in train_data.strip().split("\n"):
    train_data_list.append(i)

###############################################
             
test_labels = "testdata.label"
feLabel = open(test_labels,'r')
fedata = feLabel.read()
feLabel.close()
fedata = fedata.strip().split("\n")
te_N = len(fedata)
label_arr_te = np.zeros((te_N,D_out),dtype=float)
for i in range(te_N):
    label_arr_te[i][int(fedata[i])] = 1


    
test_data_path = "testdata.text"
fptest = open(test_data_path,"r")
test_data = fptest.read()
fptest.close()
test_data_list = []
for i in test_data.strip().split("\n"):
    test_data_list.append(i)


###############################################

######### Get word vec code ###################

w2v_model, vec = getKvecs.getKwordVecs(k="full",num_feat=D_in,train_file="./balanced_traindata.text",test_file="./testdata.text")

###############################################

#x = Variable(torch.from_numpy(vec), requires_grad=False)
#y = Variable(torch.from_numpy(label_arr_tr).type(dtype), requires_grad=False)

criterion = nn.NLLLoss()

#########################################
#   class RNN                           #
#   RNN class definintion               #
#                                       #
#########################################
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        #combined.tanh()
        hidden = self.i2h(combined)
        
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

#n_hidden = 128
rnn = RNN(D_in, H, D_out)

vec_reshaped = vec[:N].reshape(N,1,D_in)
inp = Variable(torch.from_numpy(vec_reshaped), requires_grad=False)

hidden = Variable(torch.zeros(1, H), requires_grad = True)
output, next_hidden = rnn(inp[0],hidden)

pred_array = output.data.numpy()
pred = pred_array.argmax(axis=1)

cat = np.array([2])
cat1 = Variable(torch.from_numpy(cat))

criterion(output,cat1).backward()

#################TRAINING################
#########################################
#   def oneTrainingExample              #
#   Function to return values for one   #
#   training example                    #
#                                       #
#   Input: i = int val: index of tr     #
#                      example          #
#   Returns: tensor variable            #
#########################################
def oneTrainingExample(i):
    category = int(fpdata[i])
    
    x = getKvecs.review_to_wordlist(train_data_list[i])
    line = []
    for word in x:
        line.append(getKvecs.makeFeatureVec(word,w2v_model,D_in).reshape(1,D_in))
    category_tensor = Variable(torch.LongTensor([category]), requires_grad = False)
    if(line == []):
        line = np.zeros((1,1,D_in),dtype="float32")
    
    line_tensor = Variable(torch.from_numpy(np.asarray(line)), requires_grad = False)
    return category, line, category_tensor, line_tensor

#########################################
#   def oneTestExample                  #
#   Function to return values for one   #
#   test example                        #
#                                       #
#   Input: i = int val: index of te     #
#                      example          #
#   Returns: tensor variable            #
#########################################
def oneTestExample(i):
    category = int(fedata[i])
    
    x = getKvecs.review_to_wordlist(test_data_list[i])
    line = []
    
    for word in x:
        line.append(getKvecs.makeFeatureVec(word,w2v_model,D_in).reshape(1,D_in))
    category_tensor = Variable(torch.LongTensor([category]), requires_grad = False)
    if(line == []):
        line = np.zeros((1,1,D_in),dtype="float32")
    
    line_tensor = Variable(torch.from_numpy(np.asarray(line)), requires_grad = False)
    return category, line, category_tensor, line_tensor

# If you set this too high, it might explode. If too low, it might not learn
learning_rate =1e-3
#optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(rnn.parameters(), lr=1e-3, momentum=0.9)

#########################################
#   def train                           #
#   Function to train the model for     #
#   a single training example           #
#                                       #
#   Input: training size = Either "all" #
#                          or an int val#
#          epochs = an Int val          #
#   Returns: tensor variable            #
#########################################

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        #print line_tensor.size()
        #print hidden.size()
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward(retain_graph = True)

     #Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        if type(p.grad)!=type(None):
            p.data.add_(-learning_rate, p.grad.data)

    #optimizer.step()
    
    return output, loss.data[0]

#########################################
#   def train_model                     #
#   Function to train the model with    #
#   given training size and epochs      #
#                                       #
#   Input: training size = Either "all" #
#                          or an int val#
#          epochs = an Int val          #
#   Returns: tensor variable            #
#########################################
def train_model(training_data_size="all",epochs=5):
    #all_losses = []
    if training_data_size == "all":
        training_data_size = N
    if training_data_size > N:
        return "Error: Can not train on more than %d training examples" %N
    print "Training for epochs=%d with training examples=%d" %(epochs,training_data_size)
    for i in range(epochs):
        #Acc = 0
        current_loss = 0
        for tr_iter in range(training_data_size):
            

            #tr_iter_idx = random.randint(0,N-1)
            #print tr_iter_idx
            category, line, category_tensor, line_tensor = oneTrainingExample(tr_iter)
           
                
                
            output, loss = train(category_tensor, line_tensor)
            current_loss += loss
        
            # Print iter number, loss, name and guess
            #top_n, top_i = output.data.topk(3)
            
            #if category in top_i.numpy():
            #    Acc += 1
                #print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
            
            # Add current loss avg to list of losses
            #if iter % plot_every == 0:
            #    all_losses.append(current_loss / plot_every)
            #    current_loss = 0
        print "Epoch:",i
        #print "Acc:",(float(Acc)/float(training_data_size))*100
        print "loss:",current_loss/training_data_size
        

#########################################
#   def evaluate                        #
#   Function to predict the top 3       #
#   emoji labels for the given text     #
#                                       #
#   Input:  tweet in tensor format      #
#   Returns: output object              #
#########################################
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    #for i in range(line_tensor.size()[0]):
    #print line_tensor.size()
    #print hidden.size()
    for i in range(line_tensor.size()[0]):
        #print line_tensor.size()
        #print hidden.size()
        
        output, hidden = rnn(line_tensor[i], hidden)
    
    #output, _ = rnn(line_tensor, hidden)
    #print "output-size:",output.size()
    #print "output:",output

    return output
    

#########################################
#   def predict                         #
#   Function to predict the top 3       #
#   emoji labels for the given text     #
#                                       #
#   Input:  tweet string                #
#   Returns: Top 3 emoji labels         #
#########################################

def predict(line):
    #listline = list([line])
    sent = getKvecs.review_to_sentences(line)
    linevec = getKvecs.getAvgFeatureVecs(sent, w2v_model, D_in)
    print linevec
    line_tensor = Variable(torch.from_numpy(linevec), requires_grad = False)
    #return line_tensor
    #print line_tensor.size()[0]
    res = evaluate(line_tensor)
    n,i = res.data.topk(3)
    print "For tweet:",line," we got following labels:",i
    
    
#########################################
#   def getAcc                          #
#   Function to get accuracy over the   #
#   whole test set                      #
#                                       #
#   Input:  No arguments expected       #
#   Returns: Returns accuracy value     #
#########################################
def getAcc(dataset = "test"):
    acc = 0.0
    if dataset == "test":
        startpoint  =   N
        endpoint    =   N+len(fedata)
        test_vec    = vec[startpoint:endpoint]
        for n_iter in xrange(len(test_vec)):
            category, line, category_tensor, line_tensor = oneTestExample(n_iter)
            output       = evaluate(line_tensor)
            top_n, top_i    = output.data.topk(3)
            if n_iter%1000 == 0:
                print "Processed %d out of %d records" %(n_iter,len(test_vec))
            if category in top_i.numpy():
                    acc += 1
        return (acc/len(fedata))*100
    else:
        # Finding accuracy for training set
        N=1000
        startpoint  =   0
        endpoint    =   N
        tr_vec    = vec[startpoint:endpoint]
        for n_iter in xrange(100):#len(tr_vec)):
            category, line, category_tensor, line_tensor = oneTrainingExample(n_iter)
            output       = evaluate(line_tensor)
            top_n, top_i    = output.data.topk(3)
            if n_iter%1000 == 0:
                print "Processed %d out of %d records" %(n_iter,N)
            if category in top_i.numpy():
                    acc += 1
        return (acc/N)*100

#########################################
#   Main                                #
#                                       #
#   Input:  No arguments expected       #
#   Returns: No returns                 #
#########################################
    
train_model()
print getAcc()
print getAcc("train")
