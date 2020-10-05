# LSTM_composition
Denis Paperno, 2020

Code for testing LSTM models on the "personal relations" language. Files:

- createdata.py defines the relevant class of interpreted languages with several options (branching, number of individuals, number of relations)
- data_loader.py loads data from an interpreted language and divides it into a training, validation, and test partitions. The python file contains an explanation of command line options
- LSTM_sentence_Classifier.py initializes, trains and evaluates several instances of LSTM models on randomly generated interpreted languages and writes report files and best model parameters to disk

The code for loading data and training LSTM models is based on Yu Chen Lin's code for sentiment classification, available at

https://github.com/yuchenlin/lstm_sentence_classifier
