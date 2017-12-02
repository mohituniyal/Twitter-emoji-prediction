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


N, D_in, H, D_out = 10000, 500, 128, 20

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

n_hidden = 128
rnn = RNN(D_in, H, D_out)

vec_reshaped = vec.reshape(N,1,D_in)
input = Variable(torch.from_numpy(vec_reshaped))

hidden = Variable(torch.zeros(1, H))
output, next_hidden = rnn(input[0],hidden)

pred_array = output.data.numpy()
pred = pred_array.argmax(axis=1)

