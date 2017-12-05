#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 14:49:07 2017

@author: shruthi
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import getKvecs


N, D_in, H, D_out = 10000, 512, 128, 20

######### Our code ###########

dtype = torch.FloatTensor

w2v_model, vec = getKvecs.getKwordVecs(k=N,file_name="./balanced_traindata.text")


######
train_labels = "balanced_trainlabel.label"
fpLabel = open(train_labels,'r')
fpdata = fpLabel.read()
fpLabel.close()
fpdata = fpdata.split("\n")[:N]
label_arr = np.zeros((N,D_out),dtype=float)
for i in range(N):
    label_arr[i][int(fpdata[i])] = 1
###############################


x = Variable(torch.from_numpy(vec), requires_grad=False)
y = Variable(torch.from_numpy(label_arr).type(dtype), requires_grad=False)

criterion = nn.NLLLoss()

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

#n_hidden = 128
rnn = RNN(D_in, H, D_out)

vec_reshaped = vec.reshape(N,1,D_in)
inp = Variable(torch.from_numpy(vec_reshaped),  requires_grad=False)

hidden = Variable(torch.zeros(1, H), requires_grad = True)
output, next_hidden = rnn(inp[0],hidden)

pred_array = output.data.numpy()
pred = pred_array.argmax(axis=1)

cat = np.array([2])
cat1 = Variable(torch.from_numpy(cat))

criterion(output,cat1).backward()


#########TRAINING################

def oneTrainingExample(i):
    category = int(fpdata[i])
    line = vec[i].reshape(1,D_in)
    category_tensor = Variable(torch.LongTensor([category]), requires_grad = False)
    line_tensor = Variable(torch.from_numpy(line), requires_grad = False)
    return category, line, category_tensor, line_tensor


learning_rate = 0.55 # If you set this too high, it might explode. If too low, it might not learn

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor, hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        if type(p.grad)!=type(None):
            p.data.add_(-learning_rate, p.grad.data)

    return output, loss.data[0]


all_losses = []
training_data_size = 1000
epochs = 100;
#
#def categoryFromOutput(output):
#    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
#    category_i = top_i[0][0]
#    return all_categories[category_i], category_i
for i in range(epochs):
    Acc = 0
    current_loss = 0
    for iter in range(0, training_data_size):
        category, line, category_tensor, line_tensor = oneTrainingExample(iter)
        
        #category_tensor = n
        #line_tensor = inp[0]
        output, loss = train(category_tensor, line_tensor)
        current_loss += loss
    
        # Print iter number, loss, name and guess
        if iter % 1 == 0:
            top_n, top_i = output.data.topk(3)
            if category in top_i.numpy():
                Acc += 1
            #print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
        
        # Add current loss avg to list of losses
        #if iter % plot_every == 0:
        #    all_losses.append(current_loss / plot_every)
        #    current_loss = 0
    print "Epoch:",i
    print "Acc:",(float(Acc)/float(training_data_size))*100
    print "loss:",current_loss/training_data_size