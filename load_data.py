# _*_ coding: utf-8 _*_

import os
import sys
import torch
from torch.nn import functional as F
import numpy as np
import torchtext
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe, FastText
import logging
from nltk.tokenize import word_tokenize

LOGGER = logging.getLogger("toxic_dataset")

def load_dataset(test_sen=None, batch_size=32):

    """
    tokenizer : Breaks sentences into a list of words. If sequential=False, no tokenization is applied
    Field : A class that stores information about the way of preprocessing
    fix_length : An important property of TorchText is that we can let the input to be variable length, and TorchText will
                 dynamically pad each sequence to the longest sequence in that "batch". But here we are using fi_length which
                 will pad each sequence to have a fix length of 200.
                 
    build_vocab : It will first make a vocabulary or dictionary mapping all the unique words present in the train_data to an
                  idx and then after it will use GloVe word embedding to map the index to the corresponding word embedding.
                  
    vocab.vectors : This returns a torch tensor of shape (vocab_size x embedding_dim) containing the pre-trained word embeddings.
    BucketIterator : Defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed.
    
    """
    sequence_length = 50
    TEXT = data.Field(sequential=True, tokenize=word_tokenize, lower=True, include_lengths=True, batch_first=True, fix_length=sequence_length)
    LABEL = data.LabelField(tensor_type=torch.FloatTensor)
    
    train_data, valid_data, test_data = get_dataset(TEXT, LABEL)
    print("train.fields:", train_data.fields)
    
    glove_dimensions = 300
    TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=glove_dimensions))
    #TEXT.build_vocab(train_data, vectors=FastText())
    LABEL.build_vocab(train_data)

    word_embeddings = TEXT.vocab.vectors
    print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print ("Label Length: " + str(len(LABEL.vocab)))
    print (LABEL.vocab.stoi)

    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size=batch_size, sort_key=lambda x: len(x.text), repeat=False, shuffle=True)
    vocab_size = len(TEXT.vocab)
    return TEXT, LABEL, vocab_size, word_embeddings, train_iter, valid_iter, test_iter
    

def get_dataset(TEXT, LABEL, lower=False, vectors=None):
    if vectors is not None:
        # pretrain vectors only supports all lower cases
        lower = True
    LOGGER.debug("Reading train csv file...")
    train, val = data.TabularDataset.splits(
        path='eci_data/', format='tsv', skip_header=False,
        train='train.tsv', validation='dev.tsv',
        fields=[("text", TEXT), ("label", LABEL)]
        )

    LOGGER.debug("Reading test csv file...")
    test = data.TabularDataset(
        path='eci_data/dev.tsv', format='tsv', 
        skip_header=False,
        fields=[("text", TEXT), ("label", LABEL)]
        )
    return train, val, test

if __name__ == '__main__':
    load_dataset()
