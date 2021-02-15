# LSTM_composition
Denis Paperno, 2020-2021

Code for testing LSTM models on the "personal relations" language. Files:

- createdata.py defines the relevant class of interpreted languages with several options for generatiung the language: branching, number of individuals, number of relations
- data_loader.py loads data from a randomly generated interpreted language and divides it into a training, validation, and test partitions.
- LSTM_sentence_Classifier.py initializes, trains and evaluates several instances of recurrent models on randomly generated interpreted languages and writes report files and best model parameters to disk. Parameters of models and languages can be set via command line arguments.
- experiments_code.py runs experiments for filling Table 1 and Table 2 in the paper.

The code for loading data and training recurrent models is based on Yu Chen Lin's code for sentiment classification, available at

https://github.com/yuchenlin/lstm_sentence_classifier
