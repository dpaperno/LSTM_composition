# -*- coding: utf-8 -*-
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
#torch.manual_seed(1)
#random.seed(1)

print("START")

hidden=256

class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, num_layers,bidirectional):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers=num_layers
        self.num_directions=1+int(bidirectional)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,num_layers = num_layers,bidirectional=bidirectional)
        self.hidden2label = nn.Linear(hidden_dim*self.num_directions, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        return (autograd.Variable(torch.zeros(self.num_layers*self.num_directions, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(self.num_layers*self.num_directions, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
#        print lstm_out[-1]
#        z=torch.cat([lstm_out[-1,:self.hidden_dim],lstm_out[0,self.hidden_dim:]])
#        y  = self.hidden2label(z)
        y  = self.hidden2label(lstm_out[-1])
        log_probs = F.log_softmax(y)
        return log_probs



def get_accuracy(truth, pred):
     assert len(truth)==len(pred)
     right = 0
     for i in range(len(truth)):
         if truth[i]==pred[i]:
             right += 1.0
     return right/len(truth)

def train():
    EMBEDDING_DIM = 256
    train_data, dev_data, test_data, word_to_ix, label_to_ix, complexity = data_loader.load_MR_data()
    HIDDEN_DIM = hidden
    EPOCH = 100
    best_dev_acc = 0.0
    bidirectional= "bidirectional" in sys.argv
    model = LSTMClassifier(embedding_dim=EMBEDDING_DIM,hidden_dim=HIDDEN_DIM,
                           vocab_size=len(word_to_ix),label_size=len(label_to_ix),num_layers=1,bidirectional=bidirectional)
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(),lr = 1e-3,weight_decay=9e-3)
    #optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2)
    no_up = 0
    testlengths=[len(x[0]) for x in train_data]
    def curriculum(i):
        if i < EPOCH:
        #    return 4+2*complexity*i/EPOCH
            return 2+2*i/10
        else:
            return 99999
    for i in range(EPOCH):
        train_data_filtered = [x for x in train_data if len(x[0]) < curriculum(i)]
        print([x[0] for x in train_data_filtered])
        print(len(train_data_filtered))
        random.shuffle(train_data_filtered)
        print('epoch: %d start!' % i)
        train_epoch(model, train_data_filtered, loss_function, optimizer, word_to_ix, label_to_ix, i)
        print('now best dev acc:',best_dev_acc)
        dev_acc = evaluate(model,dev_data,loss_function,word_to_ix,label_to_ix,'dev')
        test_acc = evaluate(model, test_data, loss_function, word_to_ix, label_to_ix, 'test')
        print("test accuracy:",test_acc)
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            os.system('rm mr_best_model_acc_*.model')
            print('New Best Dev!!!')
            torch.save(model.state_dict(), 'best_models/mr_best_model_acc_' + str(int(test_acc*10000)) + '.model')
            no_up = 0
        else:
            no_up += 1
            if no_up >= 22:
                break
    from collections import defaultdict
    def statsbysize(dataset):
        bysize=defaultdict(set)
        for i in dataset:
            bysize[len(i[0])].add(i)
        for c in sorted(bysize.keys()): print("length"+str(c)+": "+str(evaluate(model, bysize[c], loss_function, word_to_ix, label_to_ix, 'train')))
    print("training accuracies by size:")
    statsbysize(train_data)
    #print("test accuracies by size:")
    statsbysize(test_data)
    print("test accuracy:"+str(test_acc))
    report.append((s,test_acc))

def evaluate(model, data, loss_function, word_to_ix, label_to_ix, name ='dev'):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []

    for sent, label in data:
        truth_res.append(label_to_ix[label])
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()
        sent = data_loader.prepare_sequence(sent, word_to_ix)
        label = data_loader.prepare_label(label, label_to_ix)
        pred = model(sent)
        pred_label = pred.data.max(1)[1].numpy()
        pred_res.append(pred_label)
        # model.zero_grad() # should I keep this when I am evaluating the model?
        loss = loss_function(pred, label)
        avg_loss += loss.item()
    avg_loss /= len(data)
    acc = get_accuracy(truth_res, pred_res)
    print(name + ' avg_loss:%g %s acc:%g' % (avg_loss,name, acc ))
    return acc



def train_epoch(model, train_data, loss_function, optimizer, word_to_ix, label_to_ix, i):
    model.train()
    
    avg_loss = 0.0
    count = 0
    truth_res = []
    pred_res = []
    batch_sent = []

    for sent, label in train_data:
        truth_res.append(label_to_ix[label])
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()
        sent = data_loader.prepare_sequence(sent, word_to_ix)
        label = data_loader.prepare_label(label, label_to_ix)
        pred = model(sent)
        pred_label = pred.data.max(1)[1].numpy()
        pred_res.append(pred_label)
        model.zero_grad()
        loss = loss_function(pred, label)
        avg_loss += loss.item()
        count += 1
        if count % 500 == 0:
            print('epoch: %d iterations: %d loss :%g' % (i, count, loss.data[0]))

        loss.backward()
        optimizer.step()
    avg_loss /= len(train_data)
#    print "abc!"
#    model(data_loader.prepare_sequence(['a','b','c'], word_to_ix))
#    print "cba!"
#    model(data_loader.prepare_sequence(['c','b','a'], word_to_ix))
    print('epoch: %d done! \n train avg_loss:%g , acc:%g'%(i, avg_loss, get_accuracy(truth_res,pred_res)))

report=[]

seeds=[]

runs=10

for i in range(runs):
    seeds.append(random.randint(0,4294967296))

for s in seeds:
    torch.manual_seed(s)
    random.seed(s)
    train()

test_acc=0.0
with open('report'+'_'.join(sys.argv[1:])+"_"+str(hidden)+'dim.tsv','w') as o:
    for n in report:
        o.write(str(n[0])+"\t"+str(n[1])+"\n")
        test_acc+=n[1]

test_acc=test_acc/runs
print("Done")
print("test accuracy:")

print(test_acc)
