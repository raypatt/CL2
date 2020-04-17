#Benjamin Omar Allen, April 13, 2020
#import gensim so that you can use the word2vec constructor
#you will need to install it with the following command
#pip install --upgrade gensim
from gensim.models import Word2Vec
import sys
import re

filePath = "train.csv"
listOfTokens1 = []
listOfTokens2 = []

listOfScores = []
listOfTokens = []
print ("STARTING")
with open(filePath, 'r') as trainingData:
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

#This model is trained off of the vocabulary in the training set
model = Word2Vec(listOfTokens)

vocab = list(model.wv.vocab)

#this is what we want to feed to the model ultiamtely
input_list = []
#this is a list that keeps track of the embeddings of each word in each sentence
listOfEmbeddings = []

#loop through each sentence/score pair
def main(): 
for sentence, score in zip(listOfTokens, listOfScores):
    #loop through each word in each sentence to get the embeddings
    for word in sentence:
        if word in vocab:
            listOfEmbeddings.append(model[word])
        else:
            listOfEmbeddings.append(0)
    input_list.append([listOfEmbeddings, score])
    listOfEmbeddings = []
    
return input_list

