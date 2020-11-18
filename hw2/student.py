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

import torch
import torch.nn as nn
import torch.optim as toptim
import torch.nn.functional as F
from torchtext.vocab import GloVe
import string

# import numpy as np
# import sklearn

from config import device

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################
def prepro_word(word):
    '''
    prepro_word takes one word as input, convert all the letters to lowercase,
    remove all the punctuations and newline character and return None if the 
    word is too short or is entirely digit

    The aim of this preprocessing is to remove the "noise" in the dataset, and 
    try to make dataset more regular. For example, the net work will treat 
    "awful", "awful!" and "AWFUL" as different word, but they actually means the 
    same. 
    '''
    # convert all the letters to lower case and remove punctuations
    word = word.lower()
    word = word.translate(str.maketrans('', '', string.punctuation))
    word = word.replace('\n', '')
    # set word to empty to be removed if it's a digit
    if word.isdigit():
        word = ''

    # if a word is shorter than 2 remove it
    length = len(word)
    if length <= 2:
        word = None
    return word

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
    processed = []

    # process every word in the given sample (a review text), see details in prepro_word()
    for word in sample:
        new_word = prepro_word(word)
        if new_word != None:
            processed.append(new_word)
    return processed

def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """

    return batch
'''
choose some common meaningless words as stopwords, which can improve weighted score by 1
no two letters word like "to" and "or" etc. since those got removed during preprocessing
'''
stopWords = {'myself', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
             'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'its', 'itself',
             'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
             'these', 'those', 'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 'having',
             'does', 'did', 'doing', 'the', 'and', 'but', 'because', 'until', 'while',
             'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
             'above', 'below', 'from', 'down', 'out', 'off', 'over', 'under', 'again', 'further',
             'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
             'most', 'other', 'some', 'such', 'nor', 'only', 'own', 'same', 'than', 'too', 'very',
             'can', 'will', 'just', 'should', 'now', 'ive'}

'''
choose dim 300 to allow max features dimension in the wordVectors,
which can approximately improve weighted score by 5 compare to using dimension 50
'''
wordVectors = GloVe(name='6B', dim=300)

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

    '''
    Each ratingOutput is a tensor with size [batch_size, 2], where the 2 is the output
    size, each output cell contains the possibilities of the cell is likely to be chosen.
    Therefore we need to get cell with the max value. Same strategy applies for converting
    categoryOutput.
    '''

    # ratingOutput and categoryOutput have the same batch size, 
    # get the batch size from output dimension 0
    batch_size = ratingOutput.size()[0]

    # initialize a long tensor to predict 0 or 1
    r = torch.rand(batch_size)
    # take the higher value cell's index as predicition for each batch
    counter = 0
    for i in ratingOutput:
        if i[0] < i[1]:
            r[counter] = 1
        else:
            r[counter] = 0
        counter+=1

    # initialize a long tensor to predict 0-4
    c = torch.rand(batch_size)
    # take the max value cell's index as prediction for each batch
    counter = 0
    for i in categoryOutput:
        for j in range(5):
            if i[j] == torch.max(i):
                c[counter] = j
        counter+=1

    return r.long(), c.long()

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

    '''
    My network applies LSTM to the initial data, and then use an hidden layer with tanh
    acitivation between the LSTM and output layer to further classify the features. 
    Then use two separate output layers with log_softmax activiation to classify rating 
    and category separately. 

    The use of LSTM can improve about 15 weighted score compare to using fully connected
    layer. This makes sense because LSTM proceed the sequence word by word, and use the
    output from the previous word as part of next word's input. And LSTM has memory cells
    to preserve long term informations which can be used as each cell's input. These features
    of LSTM allows the network extract meaning not only from each word but also the 
    interralationship between the words in the sequence.
    '''
    def __init__(self):
        super(network, self).__init__()
        # choose hid_size of LSTM, slightly larger size can improve accuaracy but also increase training time
        self.hid_size = 200
        # choose how many times that the sequence will be looped
        self.layer = 2

        self.lstm = nn.LSTM(input_size = 300, hidden_size=self.hid_size, num_layers=self.layer,batch_first=True)
        self.mid = nn.Linear(self.hid_size, 100)    # intermidiate hidden layer
        self.rat = nn.Linear(100, 2)                # output layer for rating
        self.cat = nn.Linear(100, 5)                # output layer for categrory

    def forward(self, input, length):
        out, (hn, cn) = self.lstm(input)

        # out from lstm is a tensor with size [batch_size, number of hidden layer(length of sequence), hidden layer size]
        # we only want the features exracted in the last layer, so take out[:,-1,:]
        out = F.tanh(self.mid(out[:,-1,:]))
        rat_out = F.log_softmax(self.rat(out))
        cat_out = F.log_softmax(self.cat(out))

        return rat_out, cat_out

class loss(nn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    '''
    Apply nll_loss to both classification separately. Since they share the LSTM
    and a hidden layer, we need to set retain_graph for the first backward. 
    To calculate loss, simply get an average over two losses. The way of calculating
    loss won't affect accuaracy
    '''
    def __init__(self):
        super(loss, self).__init__()
        self.rat_loss = None
        self.cat_loss = None

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        self.rat_loss = F.nll_loss(ratingOutput, ratingTarget)
        self.cat_loss = F.nll_loss(categoryOutput, categoryTarget)
        return self

    def backward(self):
        self.rat_loss.backward(retain_graph=True)
        self.cat_loss.backward()
        pass

    def item(self):
        return (self.rat_loss.item()+self.cat_loss.item())/2

net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.8
batchSize = 32
'''
Although the loss can be lower with more epoche, The weighted score converges to 84 after 20 epochs
Set an appropirate learning rate to increase the training speed
'''
epochs = 1
optimiser = toptim.SGD(net.parameters(), lr=0.05)

################################################################################
###############################  Discussion   ##################################
################################################################################

'''
With all the strategies using in the above network, the weighted score can approach 84, with 
total rating 93% correct and total category 84% correct, 81% both correct. This is a reasonable
result from the data given, but there are still a few thing that could be done to improve the
model.

The first possible improvement is use stemming or lemmatization in preprocessing data. Since the
review texts are netural language, same word in different tense spells differently and is treated
as different words by the network. I have tried to remove 's' or 'es' at the end of the words, but
that doesn't really help much. Better lemmatization could be done with external package like nltk,
which can possiblely improve the performance. It is also possible to use deep learning to train a
model to help with lemmatization.

The second possible improvement is improving the method do choose the stopwords. If we visit the
all the review texts before any preprocessing, we can create a dictionary of words with appears
frequencies assign to each words, can choose the words that has too low or too high frequencies
as stop words. By this way we get a set of stopwords that more suitable for the given dataset.

However, human language is complex and it's still impossible to get 100% accuracy, and for categories
classification it's even hard to get an accuracy over 90%. For example, if we have a review like
"This place did a great job to make me sick", while the conflict meaning extract from 'great' and
'sick' could be solved to a certain extent with LSTM, but there are just not enough information given
to classify a business category. 

Natural language classification is beatiful art and there are much more things to explore.
Really enjoy this assessment!
'''