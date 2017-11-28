#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 13:16:20 2017

@author: mohituniyal
"""
from __future__ import print_function
import torch
from torch.autograd import Variable
from getKvecs import getKwordVecs
import numpy as np

dtype = torch.FloatTensor

N, D_in, H, D_out = 1000, 300, 10, 20

vec = getKwordVecs(k=N)
#vec_tensor = torch.from_numpy(vec)

######
train_labels = "traindata.labels"
fpLabel = open(train_labels,'r')
fpdata = fpLabel.read()
fpLabel.close()
fpdata = fpdata.split("\n")[:N]
label_arr = np.zeros((N,D_out),dtype=float)
for i in range(N):
    label_arr[i][int(fpdata[i])] = 1

######

#label_arr = np.array([2,17,0,18,1])

#x = torch.randn(N,D_in).type(dtype)
#x = Variable(torch.randn(N,D_in).type(dtype), requires_grad=False)
#y = Variable(torch.randn(N,D_out).type(dtype), requires_grad=False)

'''label_sparse = np.zeros((N,D_out),dtype=float)
for i in range(label_arr.shape[0]):
    label_sparse[i][label_arr[i]] = 1'''

x = Variable(torch.from_numpy(vec), requires_grad=False)
y = Variable(torch.from_numpy(label_arr).type(dtype), requires_grad=False)


w1 = Variable(torch.randn(D_in,  H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

lr = 1e-6
gold = np.zeros((N,1),dtype=dtype)
for i in range(N):
    gold[i] = (dtype(fpdata[i]))
#!!!!!!!!!!!!!!!!

for t in range(5000):
    h      = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred  =h_relu.mm(w2)
    
    
    loss = (y_pred - gold).pow(2).sum()
    #loss = 
    print (t,loss.data[0])
    
    #Backprop
    loss.backward()
    
    
    '''grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h<0] = 0
    grad_w1 = x.t().mm(grad_h)'''
    w1.data -= lr * w1.grad.data
    w2.data -= lr * w2.grad.data
    
    w1.grad.data.zero_()
    w2.grad.data.zero_()
