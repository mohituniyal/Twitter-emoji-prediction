#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 14:04:59 2017

@author: mohituniyal
"""

# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
from torch.autograd import Variable
from getKvecs import getKwordVecs
import numpy as np

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 1000, 300, 10, 20

######### Our code ###########

dtype = torch.FloatTensor

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
###############################



# Create random Tensors to hold inputs and outputs, and wrap them in Variables.
#x = Variable(torch.randn(N, D_in))
#y = Variable(torch.randn(N, D_out), requires_grad=False)

x = Variable(torch.from_numpy(vec), requires_grad=False)
y = Variable(torch.from_numpy(label_arr).type(dtype), requires_grad=False)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Variables for its weight and bias.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(size_average=True)

learning_rate = 5e-1
for t in range(2000):
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Variable of input data to the Module and it produces
    # a Variable of output data.
    y_pred = model(x)

    # Compute and print loss. We pass Variables containing the predicted and true
    # values of y, and the loss function returns a Variable containing the
    # loss.
    loss = loss_fn(y_pred, y)
    print(t, loss.data[0])

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Variables with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Variable, so
    # we can access its data and gradients like we did before.
    for param in model.parameters():
        param.data -= learning_rate * param.grad.data
        