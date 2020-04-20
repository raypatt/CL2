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
    return listOfTokens, max_length


def convert_token_to_embedding(listOfTokens, listOfScores): 
    #This model is trained off of the vocabulary in the training set
    
    embedding_length = 100
    model = Word2Vec(listOfTokens,size=embedding_length)
    
    vocab = list(model.wv.vocab)
    
    listOfTokens, max_length = convert_tokens_to_max(listOfTokens, max_sentence_length(listOfTokens))
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
        
    return input_list, input_score, vocab, model, max_length

def get_input(filePath): 
    print("Getting tokens...")
    listOfTokens, listOfScores = get_tokens(filePath)
    print("Converting tokens to embeddings...")
    input_list, input_scores, vocab, model, max_length = convert_token_to_embedding(listOfTokens, listOfScores)
    input_list = input_list[2:]
    print("Finished converted tokens to embeddings...")
    
    
    return listOfTokens[2:], input_list, input_scores, vocab, model, max_length

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
################ EVALUATION #####################################
#################################################################

def eval_to_max_length(headline, max_length): 
    if len(headline) >= max_length: 
        headline = headline[:max_length]
    else:
        while len(headline) < max_length: 
            headline.append("<PAD>")
    return headline

def tests_to_embeddings(max_length, tests, vector_model,vocab): 
    embedding_length = 100
    ret = []
    for test in tests: 
        h1 = test[0].split(" ")
        h2 = test[1].split(" ")
        
        h1 = eval_to_max_length(h1, max_length)
        h2 = eval_to_max_length(h2, max_length)
        
        tmp_h1 = []
        for word in h1:
            if word in vocab:
                tmp_h1.append(vector_model[word])
            else:
                tmp_h1.append([0]*embedding_length)
        h1 = tmp_h1
        
        tmp_h2 = []
        for word in h2:
            if word in vocab: 
                tmp_h2.append(vector_model[word])
            else: 
                tmp_h2.append([0]*embedding_length) 
        h2 = tmp_h2 
        
        h1 = torch.tensor(h1, dtype=torch.float)
        h2 = torch.tensor(h2, dtype=torch.float)
        
        
        #### Label Code
            # 0 = tie
            # 1 = h1 funnier
            # 2 = h2 funnier 
        label = test[2]
        ret.append((h1, h2, label))
    return ret

def evaluate(filePath, model, vector_model, max_length, vocab) :
    embedding_length = 100
    tests = get_test_file(filePath)
    embedded_tests = tests_to_embeddings(max_length, tests, vector_model, vocab)[2:]
    
    chunked_embedded_tests = chunks(embedded_tests, 10)
    
    
    total_correct = 0
    total = 0 
    for chunk in chunked_embedded_tests:
        c1_list = []
        c2_list = []
        chunk_labels = []
        
        if len(chunk) == 10: 
            for test in chunk: 
                c1_list.append(test[0])
                c2_list.append(test[1])
                chunk_labels.append(test[2])
            
            c1_scores = model(torch.stack(c1_list))
            c2_scores = model(torch.stack(c2_list))
            
            actual_labels = []
            for idx, c1_score in enumerate(c1_scores):
                label = 0
                if c1_score > c2_scores[idx]: 
                    label = 1
                if c1_score < c2_scores[idx]:
                    label = 2 
                actual_labels.append(label)
            
            for idx, i in enumerate(chunk_labels): 
                chunk_labels[idx] = int(i[0])
                
                
            for idx, predicted in enumerate(actual_labels): 
                actual = chunk_labels[idx]
                if predicted == actual: total_correct += 1
                total += 1
    print (total_correct/float(total))
                
        
    

def get_test_file(filePath): 
    listOfTokens1 = []
    listOfTokens2 = []
    
    items_for_evaluation = []
    
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
            
        ####Variable coding
            # 1 -- Headline 1 is funnier
            # 2 -- Headline 2 is funnier 
        #headline1_funniness = elements[4]
        #headline2_funniness = elements[8] 
        
        #funnier = 0
        #if headline1_funniness > headline2_funniness: 
        #    funnier = 1
        #else: 
        #    funnier = 2
        
        #entry = (edited_headline1, edited_headline2, funnier)
        #items_for_evaluation.append(entry)
        
        
            
        ## Old code.. potentially important? Took this from the get_inputs 
        #listOfTokens.append(listOfTokens1)
        #listOfTokens.append(listOfTokens2)
        #listOfScores.append(elements[4])
        #listOfScores.append(elements[8])
        #listOfTokens1 = []
        #listOfTokens2 = []
        label = elements[9]
        items_for_evaluation.append((edited_headline1, edited_headline2, label)) 
        
    return items_for_evaluation
    

#################################################################
################ MAIN PROCESS ###################################
#################################################################
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
def main(): 
    tokens, inputs, input_scores, vocab, vector_model, max_length = get_input("train.csv")
    
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
    
    tmp = evaluate("test.csv", model, vector_model, max_length, vocab)
    
    return model, inputs, input_scores, tmp

model, inputs, input_scores, tmp = main()
