#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 09:53:35 2020

WHENEVER USING THE TRAINING, TEST, OR DEV DATA, MUST CHANGE **ALL** COMMAS TO
A DIFFERENT TOKEN, I HAVE BEEN USING '~'. THIS IS A FAULT IN OUR PREPROCESSING
CODE-- SINCE WE ARE SEPARATING THE .CSV BY ',', DOING SO WITH OUR DATA TOTALLY
MESSES UP THE I/O. 

import gensim so that you can use the word2vec constructor
#you will need to install it with the following command
pip install --upgrade gensim
    //I installed it, but i think Anaconda includes this... -RP//
    
i had to install torchtext to get the inputs right, this was not on Anaconda
conda install -c pytorch torchtext 
conda install -c powerai sentencepiece

@author: raypatt
"""

#################################################################
############### PREPROCESSING ###################################
#################################################################

from gensim.models import Word2Vec
import sys
import re

filePath = "tmp.csv"

def get_tokens(filePath): 
    listOfTokens1 = []
    listOfTokens2 = []
    
    listOfScores = []
    listOfTokens = []
    with open(filePath, 'r', encoding = "ISO-8859-1") as trainingData:
      for line in trainingData:
        elements = line.split(',')
        edited_headline1 = re.sub(r'<[a-zA-Z]*\/>',elements[2], elements[1])
        edited_headline2 = re.sub(r'<[a-zA-Z]*\/>',elements[6], elements[5])
        for word in edited_headline1.split(' '):
            listOfTokens1.append(word)
        for word in edited_headline2.split(' '):
            listOfTokens2.append(word)
        listOfTokens.append(listOfTokens1)
        listOfTokens.append(listOfTokens2)
        listOfScores.append(elements[4])
        listOfScores.append(elements[8])
        listOfTokens1 = []
        listOfTokens2 = []
    return listOfTokens, listOfScores

def max_sentence_length(listOfTokens): 
    max_length = 0
    for sentence in listOfTokens: 
        if len(sentence)>max_length: 
            max_length = len(sentence) 
    return max_length
    
def convert_tokens_to_max(listOfTokens,max_length): 
    pad_token = "<PAD>"
    for sentence in listOfTokens: 
        if len(sentence) < max_length: 
            while len(sentence) < max_length: 
                sentence.append(pad_token)
    return listOfTokens

def convert_token_to_embedding(listOfTokens, listOfScores): 
    #This model is trained off of the vocabulary in the training set
    
    embedding_length = 100
    model = Word2Vec(listOfTokens,size=embedding_length)
    
    vocab = list(model.wv.vocab)
    
    listOfTokens = convert_tokens_to_max(listOfTokens, max_sentence_length(listOfTokens))
    #this is what we want to feed to the model ultiamtely
    input_list = []
    input_score = []
    #this is a list that keeps track of the embeddings of each word in each sentence
    listOfEmbeddings = []
    
    #remove titles
    listOfTokens = listOfTokens[2:]
    listOfScores = listOfScores[2:]
    
    #loop through each sentence/score pair
    idx = 0
    #total = len(zip(listOfTokens[2:], listOfScores[2:]))
    for idx, sentence in enumerate(listOfTokens):
        listOfEmbeddings = []
        #print idx/float(len(listOfTokens))
        score = listOfScores[idx]
    #loop through each word in each sentence to get the embeddings
        for word in sentence:
            if word in vocab:
                listOfEmbeddings.append(model[word])
                embedding_length = len(model[word])
            else:
                listOfEmbeddings.append([0]*embedding_length)
        listOfEmbeddings = torch.tensor(listOfEmbeddings, dtype=torch.float)
        try:
            input_score.append(float(score))
            input_list.append(listOfEmbeddings)
            listOfEmbeddings = []
        except:
            listOfEmbeddings = []
        
    return input_list, input_score, vocab

def get_input(filePath): 
    print("Getting tokens...")
    listOfTokens, listOfScores = get_tokens(filePath)
    print("Converting tokens to embeddings...")
    input_list, input_scores, vocab = convert_token_to_embedding(listOfTokens, listOfScores)
    input_list = input_list[2:]
    print("Finished converted tokens to embeddings...")
    
    
    return listOfTokens[2:], input_list, input_scores, vocab

#################################################################
################### CNN MODEL CLASS #############################
#################################################################
 
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout):
        
        super(CNN, self).__init__()
        
        #self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.conv_0 = nn.Conv2d(in_channels = 1, 
                                out_channels = n_filters, 
                                kernel_size = (filter_sizes[0], embedding_dim))
        
        self.conv_1 = nn.Conv2d(in_channels = 1, 
                                out_channels = n_filters, 
                                kernel_size = (filter_sizes[1], embedding_dim))
        
        self.conv_2 = nn.Conv2d(in_channels = 1, 
                                out_channels = n_filters, 
                                kernel_size = (filter_sizes[2], embedding_dim))
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, text):
                
        #text = [batch size, sent len]
        
        #embedded = self.embedding(text)
                
        #embedded = [batch size, sent len, emb dim]
        
        embedded = text.unsqueeze(1)
        
        #embedded = [batch size, 1, sent len, emb dim]
        conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
        conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = F.relu(self.conv_2(embedded).squeeze(3))
        
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        
        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim = 1))

        #cat = [batch size, n_filters * len(filter_sizes)]
        
        fc = self.fc(cat)
        
        return fc
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#################################################################
################### TRAIN CNN MODEL #############################
#################################################################

def binary_accuracy(predictions, outputs): 
    val = True
    for idx, i in enumerate(predictions): 
        tmp = [outputs[idx]-.1, outputs[idx]+.1]
        if (i < tmp[0]) | (i > tmp[1]): 
            val = False
    return val
        

def train(model, inputs, outputs, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    correct = 0
    total = 0
    
    input_size = 0
    
    for idx, input_batch in enumerate(inputs): 
        if idx == 0: 
            input_size = len(input_batch)
            
        output_batch = outputs[idx]
        
        predictions = model(input_batch)
        
        if len(input_batch) == input_size: 
            predictions = torch.reshape(predictions, [len(input_batch),1])
            output_batch = torch.reshape(output_batch, [len(input_batch),1])
            
        
            if binary_accuracy(predictions, output_batch):
                correct += 1
            total +=1
                
            loss = criterion(predictions, output_batch)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
    print(correct/float(total))
    print("correct: ",correct)
    print("total: ",total)
    
    return epoch_loss / len(inputs)
        
    
    #for batch in iterator:
        
    #    optimizer.zero_grad()
        
    #    input = batch[0]
    #    output = batch[1]
        
    #    predictions = model(input).squeeze(1)
    #    print predictions
    #    loss = criterion(predictions, output)
        
    #    acc = binary_accuracy(predictions, output)
        
    #    loss.backward()
        
    #    optimizer.step()
        
    #    epoch_loss += loss.item()
    #    epoch_acc += acc.item()
        
    #return epoch_loss / len(iterator), epoch_acc / len(iterator)

#################################################################
################ MAIN PROCESS ###################################
#################################################################
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
def main(): 
    tokens, inputs, input_scores, vocab = get_input("tmp2.csv")
    
    input_scores = torch.tensor(input_scores, dtype=torch.float)
    
    print("Doing setup...")
    INPUT_DIM = max_sentence_length(tokens)
    EMBEDDING_DIM = 100
    N_FILTERS = 100
    FILTER_SIZES = [2,3,4]
    OUTPUT_DIM = 1
    DROPOUT = 0.5
    #PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    
    ### Condense inputs into Tensor [ #Examples, Sentence_Length, EmbeddingSize]
    inputs = torch.stack(inputs)
    
    ###split inputs and input_scores into X groups..
    print("Make batches...")
    inputs = torch.split(inputs, 10)
    input_scores = torch.split(input_scores, 10)
    
    print("Make model..")
    model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
    
    import torch.optim as optim
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)
    
    
    N_EPOCHS = 50

    best_valid_loss = float('inf')
    
    for epoch in range(N_EPOCHS):
        print("Epoch: " + str(epoch))
        train_loss = train(model, inputs, input_scores, optimizer, criterion)
        print(train_loss)
        
    return model, inputs, input_scores

model, inputs, input_scores = main()
