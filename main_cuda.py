import pickle
import copy
import os
import time
import load_data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
#from models.LSTM import LSTMClassifier
#from models.LSTM_Attn import AttentionModel
from models.RCNN import RCNN, RCNN3Layers
#from models.selfAttention import SelfAttention
#from sklearn.model_selection import ParameterGrid

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)
    
def train_model(model, train_iter, epoch, optim, loss_fn, batch_size):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.cuda()
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        text = batch.text[0]
        target = batch.label
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()
        if (text.size()[0] is not batch_size):# One of the batch returned by BucketIterator has length different than 32.
            continue
        optim.zero_grad()
        prediction = model(text)
        loss = loss_fn(prediction, target)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/len(batch)
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1
        
        #if steps % 100 == 0:
        #    print ('Epoch: {}, Idx: {}, Training Loss: {}, Training Accuracy: {}%'.format(epoch+1, idx+1, loss.item(), acc.item()))
        
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
        
    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)

def eval_model(model, val_iter, loss_fn, batch_size):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text = batch.text[0]
            if (text.size()[0] is not batch_size):
                continue
            target = batch.label
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                text = text.cuda()
                target = target.cuda()
            prediction = model(text)
            loss = loss_fn(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects/len(batch)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter)
	

batch_size = 32
output_size = 3 # n etiquetas
hidden_size = 2048
embedding_length = 300
epochs = 10
learning_rate = 1e-4

TEXT, LABEL, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_data.load_dataset(batch_size=batch_size)

#model = LSTMClassifier(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
#model = AttentionModel(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
model = RCNN(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
#model = SelfAttention(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
#model = RCNN3Layers(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)

optim = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
loss_fn = F.cross_entropy

if __name__ == '__main__':
    last_val_acc = 0
    #last_model = 0
    for epoch in range(epochs):
        model.zero_grad()
        train_loss, train_acc = train_model(model, train_iter, epoch, optim, loss_fn, batch_size)
        val_loss, val_acc = eval_model(model, valid_iter, loss_fn, batch_size)
        print('Epoch: {}, Train Loss: {}, Train Acc: {}%, Val. Loss: {}, Val. Acc: {}%'.format(epoch, train_loss, train_acc, val_loss, val_acc))
        if val_acc > last_val_acc:
            last_val_acc = val_acc
            last_model = copy.copy(model)
        else:
            print('Early stopping!')
            model = last_model
            break
    test_loss, test_acc = eval_model(model, test_iter, loss_fn, batch_size)
    print('Test Loss: {}, Test Acc: {}%'.format(test_loss, test_acc))
    # save
    modelStateDict = str(round(test_acc, 2)) + str(model)[:4] +  str(learning_rate)
    torch.save(model.state_dict(), './modelDicts/' + modelStateDict + '.msd')
    string = '{}_{}_{}_{}_{}'.format(str(round(test_acc, 2)), str(model)[:4], batch_size, hidden_size, learning_rate)
    pickle.dump(string, open('./modelDicts/' + string + '.dict', 'wb'))
    