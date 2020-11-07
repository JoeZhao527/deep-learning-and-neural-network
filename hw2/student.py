#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables stopWords,
wordVectors, trainValSplit, batchSize, epochs, and optimiser, as well as a basic
tokenise function.  You are encouraged to modify these to improve the
performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

# import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as toptim
from torchtext.vocab import GloVe
# import numpy as np
# import sklearn

from config import device

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################

def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """

    processed = sample.split()

    return processed

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """

    return sample

def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """

    return batch

stopWords = {}
wordVectors = GloVe(name='6B', dim=50)

################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.
    """
    s = ratingOutput.size()[0]
    r = torch.rand(s)
    counter = 0
    for i in ratingOutput:
        if i[0] > i[1]:
            r[counter] = 0
        else:
            r[counter] = 1
        counter+=1

    c = torch.rand(s)
    counter = 0
    for i in categoryOutput:
        for j in range(5):
            if i[j] == torch.max(i):
                c[counter] = j
        counter+=1
    return r, c

################################################################################
###################### The following determines the model ######################
################################################################################

class network(nn.Module):
    """
    Class for creating the neural network.  The input to your network will be a
    batch of reviews (in word vector form).  As reviews will have different
    numbers of words in them, padding has been added to the end of the reviews
    so we can form a batch of reviews of equal length.  Your forward method
    should return an output for both the rating and the business category.
    """

    def __init__(self):
        super(network, self).__init__()
        self.cat_in = nn.Linear(50,70)
        self.cat_out = nn.Linear(70,5)

        self.rat_in = nn.Linear(50,70)
        self.rat_out = nn.Linear(70,2)

    def forward(self, input, length):
        input = torch.sum(input, dim=1)
        input = torch.transpose(input,0,1)
        input = torch.div(input, length)
        input = torch.transpose(input,0,1)

        r = F.tanh(self.rat_in(input))
        r = F.log_softmax(self.rat_out(r))

        c = F.tanh(self.cat_in(input))
        c = F.log_softmax(self.cat_out(c))
        return r,c

class loss(nn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()
        self.r = None
        self.c = None

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        self.r = F.nll_loss(ratingOutput,ratingTarget)
        self.c = F.nll_loss(categoryOutput,categoryTarget)
        return self
    
    def backward(self):
        self.r.backward()
        self.c.backward()
        pass
net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.8
batchSize = 32
epochs = 10
optimiser = toptim.SGD(net.parameters(), lr=0.01)
