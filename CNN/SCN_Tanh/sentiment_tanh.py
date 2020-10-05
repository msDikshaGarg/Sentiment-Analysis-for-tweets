

from __future__ import division
import os
import pickle
import numpy as np
from sknn.mlp import Convolution, Layer, Classifier, Regressor
from glove_encoder import GloveEncoder
from sstb import SSTb
import math, re, string, requests, json
from itertools import product
from inspect import getsourcefile
from os.path import abspath, join, dirname
import csv
import cgi

def batch_train(train,val,model_path):
    trainX,trainY = train
    valX,valY = val
    nn = Classifier(layers = [
			Convolution('Tanh',
                                    channels=100,
                                    kernel_shape=(5,WORD_DIM),
                                    border_mode='valid'
                                    #pool_shape=(3,1),
                                    #pool_type='max'
                                    ),
			Layer('Tanh',units=900,dropout=0.5),
                        Layer('Softmax')],
                        batch_size = 50,
                        learning_rate = 0.02, ###Change here: 1
                        normalize='dropout',
                        verbose = True)
    nn.n_iter = 300 ###Change here -3 (value changed from 100 to 3)
    print 'Net created...'
    try:
	nn.fit(trainX,trainY)
    except KeyboardInterrupt:
	pickle.dump(nn,open(model_path,'wb'))
    pickle.dump(nn,open(model_path,'wb'))
    print 'Done, final model saved'
    print 'Testing'
    #Accuracy on the validation set (Library out: N/A)
    print 'Validation accuracy:',batch_test(model_path,val)

def batch_test(model_path,test):
        testX,testY = test
        nn = pickle.load(open(model_path,'rb'))
	Y = nn.predict(testX)
	correct = 0
	total = 0
	print Y.shape,'==',testY.shape
	for i,y in enumerate(Y):
            good = True
            for j in range(len(y)):
                if y[j] != testY[i][j]:
                    good = False
                    break
            if good:
		correct = correct+1
	    total = total+1
	return (correct/total)

#Interactive test/eval
def interactive_test(path):



    
    def csv_dict_reader(file_obj):
        nn = pickle.load(open(model_path,'rb'))
        glove = GloveEncoder(glove_path,WORD_DIM,PHRASE_LEN)
        reader = csv.DictReader(file_obj)
        for lin in reader:
            
            
            line=lin['text'].strip()
            print line
            em = glove.encode_phrase(line)
            
            p = nn.predict(np.array([em]))[0]
            
            if p[0] == 1:
                    print 'Negative'
            elif p[1] == 1: ### Changes here -2 (deleted tabs)
                    print 'Neutral'
            else:
                    print 'Positive' ### to here -2
    with open("../realDonaldTrump_tweets2.csv") as f_obj:
        csv_dict_reader(f_obj)
        


#Driver
sstb_path = 'SSTb'
glove_path = 'glove'
model_path = 'sentiment_model.pkl'

#Parameters to encode text with GloVe vectors (see glove_encoder.py)
WORD_DIM = 50
PHRASE_LEN = 15
#Parameters to load dataset
COUNT = 20000
MAX_NEG = 0.45
MIN_POS = 0.65
TRAIN_FRAC = 0.6        #Fraction of the dataset for the training set
VAL_FRAC = 0.2          #Fraction of the training set for the validation set
                        #The remaining 0.2 will be the test set

#Load glove encoder
glove =	GloveEncoder(glove_path,WORD_DIM,PHRASE_LEN)
#Load SSTb dataset
sstb = SSTb(MAX_NEG, MIN_POS, glove, sstb_path)
train,val,test = sstb.load_ds(COUNT, TRAIN_FRAC,VAL_FRAC)
#Train with mini batches SGD
batch_train(train,val,model_path)
#Print accuracy on the test set
print 'Test accuracy: '+str(batch_test(model_path,test))
#Shell-like interface to test with manual input
interactive_test(model_path)
