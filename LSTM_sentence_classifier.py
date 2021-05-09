# -*- coding: utf-8 -*-
# code for training and testing recurrent models on interpreted languages
# Denis Paperno, 2018-2021
import sys
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_loader
import os
import random
torch.set_num_threads(8)
import argparse
from collections import defaultdict

def gentle_curriculum(i,num_epochs):
    if i < num_epochs:
        return 2+2*i/10
    else:
        return 99999

def slow_curriculum(i,num_epochs):
    if i < num_epochs:
        return 2+i/10
    else:
        return 99999

def steep_curriculum(i,num_epochs):
    if i < num_epochs and i < 20:
        return 2+2*i/10
    else:
        return 99999

def no_curriculum(i,num_epochs): return 99999

lookup={"no_curriculum":no_curriculum,"gentle_curriculum":gentle_curriculum,"steep_curriculum":steep_curriculum,"slow_curriculum":slow_curriculum}

class LSTMClassifier(nn.Module):
    """Class of recurrent sequence classifiers"""
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, num_layers,bidirectional,rnn="LSTM"):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers=num_layers
        self.num_directions=1+int(bidirectional)
        self.bidirectional = bidirectional
        self.architecture = rnn
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if rnn=="LSTM": rnnmodel = nn.LSTM(embedding_dim, hidden_dim,num_layers = num_layers,bidirectional=bidirectional)
        elif rnn=="GRU": rnnmodel = nn.GRU(embedding_dim, hidden_dim,num_layers = num_layers,bidirectional=bidirectional)
        else: rnnmodel = nn.RNN(embedding_dim, hidden_dim,num_layers = num_layers,bidirectional=bidirectional)
        self.lstm = rnnmodel
        self.hidden2label = nn.Linear(hidden_dim*(self.num_directions), label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        if self.architecture=="LSTM": 
            hidden_layer = (autograd.Variable(torch.zeros(self.num_layers*self.num_directions, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(self.num_layers*self.num_directions, 1, self.hidden_dim)))
        else: hidden_layer = autograd.Variable(torch.zeros(self.num_layers*self.num_directions, 1, self.hidden_dim))
        return hidden_layer

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        if self.bidirectional: z=torch.cat((lstm_out[-1,:self.hidden_dim],lstm_out[0,self.hidden_dim:]),0)
        else: z=lstm_out[-1]
        y  = self.hidden2label(z)
#        y  = self.hidden2label(torch.cat([lstm_out[0],lstm_out[-1]]))
#        y  = self.hidden2label(lstm_out[-1])
        log_probs = F.log_softmax(y)
        return log_probs



def get_accuracy(truth, pred):
     assert len(truth)==len(pred)
     assert len(truth)>0
     right = 0
     for i in range(len(truth)):
         if truth[i]==pred[i]:
             right += 1.0
     return right/len(truth)

def train(params,report,seed,debug=False):
    """Action happens here for a single run. Data is generated, a recurrent model is initialized, trained and evaluated."""
    bidirectional=params.bidirectional
    rev=params.rev
    architecture=params.architecture
    curriculum=lookup[params.curriculum]
    EMBEDDING_DIM = params.emb_dim
    EPOCH = params.epochs
    HIDDEN_DIM = params.hidden
    train_data, dev_data, test_data, word_to_ix, label_to_ix, complexity = data_loader.load_data(params)
    best_dev_acc = 0.0
    model = LSTMClassifier(embedding_dim=EMBEDDING_DIM,hidden_dim=HIDDEN_DIM,
                           vocab_size=len(word_to_ix),label_size=len(label_to_ix),num_layers=1,bidirectional=bidirectional,rnn=architecture)
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(),lr = 1e-3,weight_decay=9e-3)
    no_up = 0
    testlengths=[len(x[0]) for x in train_data]
    for i in range(EPOCH):
        train_data_filtered = [x for x in train_data if len(x[0])-x[0].count("the") < curriculum(i,EPOCH)]
        if debug: print([x[0] for x in train_data_filtered])
        if debug: print(len(train_data_filtered))
        random.shuffle(train_data_filtered)
        print('epoch %d starts!' % i)
        train_epoch(model, train_data_filtered, loss_function, optimizer, word_to_ix, label_to_ix, i,rev=rev)
        print('now best dev acc:',best_dev_acc)
        dev_acc = evaluate(model,dev_data,loss_function,word_to_ix,label_to_ix,'dev',rev=rev)
        test_acc = evaluate(model, test_data, loss_function, word_to_ix, label_to_ix, 'test',rev=rev)
        print("test accuracy:",test_acc)
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            if debug: 
                os.system('rm best_models/best_model_acc_*.model')
                print('New Best Dev!!!')
                torch.save(model.state_dict(), 'best_models/best_model_acc_' + str(int(test_acc*10000)) + '.model')
            no_up = 0
        else:
            no_up += 1
            if no_up >= 22:
                break
    
    def statsbysize(dataset):
        bysize=defaultdict(set)
        for i in dataset:
            bysize[len(i[0])].add(i)
        for c in sorted(bysize.keys()): print("length"+str(c)+": "+str(evaluate(model, bysize[c], loss_function, word_to_ix, label_to_ix, 'train', rev=rev)))
    if debug: 
        print("training accuracies by size:")
        statsbysize(train_data)
        print("test accuracies by size:")
        statsbysize(test_data)
        print("test accuracy:"+str(test_acc))
    report.append((seed,test_acc))

def evaluate(model, data, loss_function, word_to_ix, label_to_ix, name ='dev', rev=False):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []

    for sent, label in data:
        truth_res.append(label_to_ix[label])
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()
        sent = data_loader.prepare_sequence(sent, word_to_ix, rev=rev)
        label = data_loader.prepare_label(label, label_to_ix)
        pred = model(sent)
        pred_label = pred.data.max(1)[1].numpy()
        pred_res.append(pred_label)
        loss = loss_function(pred, label)
        avg_loss += loss.item()
    avg_loss /= len(data)
    acc = get_accuracy(truth_res, pred_res)
    print(name + ' avg_loss:%g %s acc:%g' % (avg_loss,name, acc ))
    return acc



def train_epoch(model, train_data, loss_function, optimizer, word_to_ix, label_to_ix, i,rev=False,debug=False):
    model.train()
    
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    batch_sent = []

    for sent, label in train_data:
        truth_res.append(label_to_ix[label])
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()
        sent = data_loader.prepare_sequence(sent, word_to_ix,rev=rev)
        label = data_loader.prepare_label(label, label_to_ix)
        pred = model(sent)
        pred_label = pred.data.max(1)[1].numpy()
        pred_res.append(pred_label)
        model.zero_grad()
        loss = loss_function(pred, label)
        avg_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_loss /= len(train_data)
    if debug: print('epoch: %d done! \n train avg_loss:%g , acc:%g'%(i, avg_loss, get_accuracy(truth_res,pred_res)))

parser = argparse.ArgumentParser(description='creates an interpreted language of personal relations and trains and evaluates a recurrent model on it')
#set command line parameters and their default values:
parser.add_argument('--num_pairs', dest='num_pairs', type=int, default=2)
parser.add_argument('--num_rels', dest='rel_num', type=int, default=2)
parser.add_argument('-c', dest='complexity', type=int, default=3)
parser.add_argument('--min_c', dest='min_complexity', type=int, default=parser.parse_args().complexity)
parser.add_argument('-b', dest='branching', type=str, default="l")
parser.add_argument('--top_complexity_in_train', dest='top_complexity_share_in_training', type=float, default=0.8)
parser.add_argument('--rev', dest='rev', type=bool, default=False)
parser.add_argument('--cur', dest='curriculum', type=str, default="gentle_curriculum")
parser.add_argument('--arch', dest='architecture', type=str, default="LSTM")
parser.add_argument('--bidir', dest='bidirectional', type=bool, default=False)
parser.add_argument('--embdim', dest='emb_dim', type=int, default=256)
parser.add_argument('--epochs', dest='epochs', type=int, default=100)
parser.add_argument('--hidden', dest='hidden', type=int, default=256)
parser.add_argument('--runs', dest='runs', type=int, default=10)
parser.add_argument('-o', dest='outfile', type=str, default="report.tsv")
params = parser.parse_args()

def run_with(params):
    """run a single parameter setting, incl. a specified number of runs, and write to report"""
    report=[]
    seeds=[]
    runs=params.runs
    for i in range(runs):
        seeds.append(random.randint(0,4294967296))

    with open(params.outfile,'a') as o:
        o.write("starting "+str(params.runs)+" runs with parameters:\n")
        o.write(str(params)+"\n")
        o.write("detailed report, random seed vs. test accuracy:\n")
    
    for s in seeds:
        torch.manual_seed(s)
        random.seed(s)
        train(params,report,s)
        with open(params.outfile,'a') as o:
            o.write(str(s)+"\t"+str(report[-1][1])+"\n")
    
    test_acc=0.0
    perf_acc=0.0
    for n in report:
        test_acc+=n[1]
        perf_acc+=float(n[1]==1.0)
    
    test_acc=test_acc/runs
    perf_acc=perf_acc/runs
    print("test accuracy:",test_acc)
    print("perfect accuracy percentage:",perf_acc)
    
    with open(params.outfile,'a') as o:
        o.write("finished "+str(params.runs)+" runs with parameters:\n")
        o.write(str(params)+"\n")
        o.write("avg test accuracy: "+str(test_acc)+"\n")
        o.write("perfect test accuracy percentage: "+str(perf_acc)+"\n")

if __name__ == "__main__":
    run_with(params)
