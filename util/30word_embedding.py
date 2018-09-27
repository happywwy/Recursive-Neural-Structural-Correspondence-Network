# -*- coding: utf-8 -*-
"""
Created on Wed May 27 19:50:17 2015

@author: happywwy1991
"""
"""
convert pre-trained word embeddings to input vector for each word
create dictionary
Pre-trained word embeddings are better to be normalized
"""

import numpy as np
import cPickle

# replace this file with the pre-trained word embeddings of your own
dic_file = open("/home/wenya/Word2Vec_python_code_data/data/w2v_merge_norm100_10.txt", "r")
dic = dic_file.readlines()
dictionary = {}

for line in dic:
    word_vector = line.split(",")[:-1]
    vector_list = []
    for element in word_vector[len(word_vector)-100:]:
        vector_list.append(float(element))
    word = ','.join(word_vector[:len(word_vector)-100])
  
    vector = np.asarray(vector_list)
    dictionary[word] = vector# / np.linalg.norm(vector)

# load input file    
final_input = cPickle.load(open("data_semEval/final_input_lapdev_split4", "rb"))
vocab = final_input[0]
# store dictionary into word_embedding
word_embedding = np.zeros((100, len(vocab)))
count = 0

for ind, word in enumerate(vocab):
    if word in dictionary.keys():
        vec = dictionary[word]
        row = 0
        for num in vec:
            word_embedding[row][ind] = float(num)
            row += 1
        count += 1
    # if the word is not in the vocab, randomly initialize it
    else:
        print word,
        for i in range(100):
            word_embedding[i][ind] = 2 * np.random.rand() - 1
        word_embedding[:,ind] = word_embedding[:,ind] / np.linalg.norm(word_embedding[:,ind])
    
print len(vocab)
print count
cPickle.dump(word_embedding, open("data_semEval/word_embeddings100_lapdev_norm", "wb"))
