# -*- coding: utf-8 -*-
# loads interpreted language data folowing set parameters.
import torch
import torch.autograd as autograd
import codecs
import random
import torch.utils.data as Data
import createdata

SEED = 20034

def prepare_sequence(seq, to_ix, cuda=False, rev=False):
    """encodes and possibly reverse a sequence of tokens
    input: a sequence of tokens, and a token_to_index dictionary
    output: a LongTensor variable to encode the sequence of idxs

    """
    if rev: seq=reversed(seq)
    var = autograd.Variable(torch.LongTensor([to_ix[w] for w in seq]))
    return var

def prepare_label(label,label_to_ix, cuda=False):
    """encode the label of the sequence"""
    var = autograd.Variable(torch.LongTensor([label_to_ix[label]]))
    return var

def build_token_to_ix(sentences):
    """build a token-to-index dictionary from a list of sentences"""
    token_to_ix = dict()
    print(len(sentences))
    for sent in sentences:
        for token in sent:
            if token not in token_to_ix:
                token_to_ix[token] = len(token_to_ix)
    token_to_ix['<pad>'] = len(token_to_ix)
    return token_to_ix

def build_label_to_ix(labels):
    """build a label-to-index dictionary from a list"""
    label_to_ix = dict()
    for label in labels:
        if label not in label_to_ix:
            label_to_ix[label] = len(label_to_ix)


def load_data(params):
    """returns training, validation and test data from an interpreted language following parameters in a namespace"""
    print('generating data')
    #read individual parameter: number of entity pairs
    num_pairs=params.num_pairs
    #read individual parameters: number of relation names
    rel_num=params.rel_num
    #create a random interpreted language with given vocabulary size parameters
    L=createdata.InterpretedLanguage(rel_num,num_pairs)
    k=num_pairs*5*rel_num
    
    branching=params.branching
    min_complexity=params.min_complexity
    complexity=params.complexity
    thedata = L.allexamples(branching,complexity=min_complexity-1)

    random.seed(SEED)
    random.shuffle(thedata)
    
    #as development and test data, we use all examples of the highest complexity
    high_complexity_data=L.allexamples(branching,complexity=complexity,min_complexity=min_complexity)
    random.shuffle(high_complexity_data)
    datasize=len(high_complexity_data)
    
    p=params.top_complexity_share_in_training
    #adding a shatre of the highest complexity data to the training partition
    train_data = thedata+high_complexity_data[:int(datasize*p)]
    #splitting parts of the remaining highest complexity data into validation and test partitions
    dev_data = high_complexity_data[int(datasize*p):int(datasize*(p+(1-p)*0.55))]
    test_data = high_complexity_data[int(datasize*(p+(1-p)*0.55)):]
    print('total data: %s; train: %s; dev: %s; test: %s' %(datasize+len(thedata),len(train_data),len(dev_data),len(test_data)))

    random.shuffle(train_data)
    random.shuffle(dev_data)
    random.shuffle(test_data)
    #print the sizes of train, validation and test partitions
    print('train:',len(train_data),'dev:',len(dev_data),'test:',len(test_data))

    word_to_ix = build_token_to_ix([s for s,_ in train_data+dev_data+test_data])
    label_to_ix = {val:idx for idx,val in enumerate(L.names)}
    print('vocab size:',len(word_to_ix),'label size:',len(label_to_ix))
    print('loading data done!')
    return train_data,dev_data,test_data,word_to_ix,label_to_ix,complexity
