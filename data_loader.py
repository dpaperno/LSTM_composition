# -*- coding: utf-8 -*-
# Command line arguments, in this order:
# * number of individual pairs in the model
# * number of relations in the model
# * n - maximal complexity of examples
# * branching type (l,r,rl)
# * (optional) proportion of examples of complexity n included in training data
# * (optional) "rev" if we want to reverse the strings of the language 
import sys
import torch
import torch.autograd as autograd
import codecs
import random
import torch.utils.data as Data
import createdata

SEED = 20034

# input: a sequence of tokens, and a token_to_index dictionary
# output: a LongTensor variable to encode the sequence of idxs
def prepare_sequence(seq, to_ix, cuda=False):
    if len(sys.argv)>6:
        if sys.argv[6]=='rev': seq=reversed(seq)
    var = autograd.Variable(torch.LongTensor([to_ix[w] for w in seq]))
    return var

def prepare_label(label,label_to_ix, cuda=False):
    var = autograd.Variable(torch.LongTensor([label_to_ix[label]]))
    return var

def build_token_to_ix(sentences):
    token_to_ix = dict()
    print(len(sentences))
    for sent in sentences:
        for token in sent:
            if token not in token_to_ix:
                token_to_ix[token] = len(token_to_ix)
    token_to_ix['<pad>'] = len(token_to_ix)
    return token_to_ix

def build_label_to_ix(labels):
    label_to_ix = dict()
    for label in labels:
        if label not in label_to_ix:
            label_to_ix[label] = len(label_to_ix)


def load_MR_data():

    print('generating data')
    num_pairs=int(sys.argv[1])
    rel_num=int(sys.argv[2])
    L=createdata.InterpretedLanguage(rel_num,num_pairs)
    k=num_pairs*5*rel_num
    
    branching=sys.argv[4]
    complexity=int(sys.argv[3])
    thedata = L.allexamples(branching,complexity=complexity-1)#L.randomexamples(k,branching,complexity=2)

    random.seed(SEED)
    random.shuffle(thedata)
    datasize=len(thedata)
    
    devtest=L.allexamples(branching,complexity=complexity,min_complexity=complexity)
    random.shuffle(devtest)
    datasize=len(devtest)
    #datasize=len(thedata)
    
    if len(sys.argv)>5: p=float(sys.argv[5])
    else: p=0.8
    #train_data = L.allexamples(branching,complexity=2)+devtest[:int(datasize*0.5)]#
    train_data = thedata+devtest[:int(datasize*p)]
    #dev_data = devtest[int(datasize*0.5):int(datasize*0.6)]
    dev_data = devtest[int(datasize*p):int(datasize*(p+(1-p)*0.55))]
    #test_data = devtest[int(datasize*0.6):]
    test_data = devtest[int(datasize*(p+(1-p)*0.55)):]#L.allexamples(branching,complexity=6,min_complexity=6)
    print('total data: %s; train: %s; dev: %s; test: %s' %(datasize+len(thedata),len(train_data),len(dev_data),len(test_data)))

    random.shuffle(train_data)
    random.shuffle(dev_data)
    random.shuffle(test_data)

    print('train:',len(train_data),'dev:',len(dev_data),'test:',len(test_data))

    word_to_ix = build_token_to_ix([s for s,_ in train_data+dev_data+test_data])
    label_to_ix = {val:idx for idx,val in enumerate(L.names)}
    print('vocab size:',len(word_to_ix),'label size:',len(label_to_ix))
    print('loading data done!')
    return train_data,dev_data,test_data,word_to_ix,label_to_ix,complexity


def load_MR_data_batch():

    pass
