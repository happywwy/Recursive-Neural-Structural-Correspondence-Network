# -*- coding: utf-8 -*-
"""
Created on Jan 22 10:28:35 2017

@author: wangwenya
"""

"""
create tree structures from raw parses for training sentences
accumulate vocabulary
ignore lemmatization
differentiate beginning and inside of aspects/opinions with ground-truth labels
"""


from dtree_util import *
import sys, cPickle, random
from numpy import *

import nltk
from nltk.corpus import wordnet as wn
import networkx as nx

# import tree files from both source domain and target domain
f_source = open('data_semEval/raw_parses_laptop', 'r')
f_target = open('data_semEval/raw_parses_device', 'r')


indice = 0
data_source = f_source.readlines()
data_target = f_target.readlines()
f_source.close()
f_target.close()

plist = []
tree_dict = []
vocab = []
rel_list = []
tree_source = []
tree_target = []
# aspect word label files
label_file_source = open('data_semEval/aspect_op_laptop', 'r')
label_file_target = open('data_semEval/aspect_op_device', 'r')
# opinion word label files
label_sentence_source = open('data_semEval/addsenti_laptop', 'r')
label_sentence_target = open('data_semEval/addsenti_device', 'r')

# process source tree files
for line in data_source:
    # construct dependency tree structure for each sentence
    if line.strip():
        rel_split = line.split('(')
        rel = rel_split[0]
        deps = rel_split[1][:-1]
        deps = deps.replace(')','')
        if len(rel_split) != 2:
            print 'error ', rel_split
            sys.exit(0)

        else:
            dep_split = deps.split(',')
            
        if len(dep_split) > 2:
            fixed = []
            half = ''
            for piece in dep_split:
                piece = piece.strip()
                if '-' not in piece:
                    half += piece

                else:
                    fixed.append(half + piece)
                    half = ''

            dep_split = fixed

        final_deps = []
        for dep in dep_split:
            words = dep.split('-')
            word = words[0]
            ind = int(words[len(words) - 1])

            if len(words) > 2:
                word = '-'.join([w for w in words[:-1]])

            final_deps.append( (ind, word.strip()) )
        # store relations and word with indices    
        plist.append((rel,final_deps))

    # end of a sentence, store all the dependencies into tree format
    else:
        edges=[]
        max_ind = -1
        for rel, deps in plist:
            for ind, word in deps:
                if ind > max_ind:
                    max_ind = ind
        # load words into nodes, then make a dependency tree
        nodes = [None for i in range(0, max_ind + 1)]
        for rel, deps in plist:
            for ind, word in deps:
                nodes[ind] = word
        tree = dtree(nodes)

        # label the words in each sentence with ground-truth labels
        aspect_term = label_file_source.readline().strip()
        labeled_sent = label_sentence_source.readline().strip()    
            
        #facilitate bio notation
        aspect_BIO = {}
        # if opinion words exist
        if '##' in labeled_sent:
                opinions = labeled_sent.split('##')[1].strip()
                opinions = opinions.split(',')                
                for opinion in opinions:
                    if opinion != '':
                        op_list = opinion.split()[:-1]
                        pol = opinion.split()[-1]
                        if len(op_list) > 1:
                            for ind, term in enumerate(nodes):
                                if term != None:
                                    if term.lower() == op_list[0].lower() and ind < len(nodes) - 1 and nodes[ind + 1] != None\
                                      and nodes[ind + 1].lower() == op_list[1].lower():
                                        tree.get(ind).trueLabel = 3
                                        for i in range(len(op_list) - 1):
                                            if nodes[ind + i + 1].lower() == op_list[i + 1].lower():
                                                tree.get(ind + i + 1).trueLabel = 4
                                            
                        elif len(op_list) == 1:
                            for ind, term in enumerate(nodes):
                                if term != None:
                                    if term.lower() == op_list[0].lower() and tree.get(ind).trueLabel == 0:
                                        tree.get(ind).trueLabel = 3
        # if aspect words exist
        if aspect_term != 'NIL':
            aspects = aspect_term.split(',')
            for aspect in aspects:
                aspect = aspect.strip()
                target = aspect.split(':')[0]
                #aspect is a phrase
                if ' ' in target:
                    aspect_list = target.split()
                    for ind, term in enumerate(nodes):
                        if term != None and ind < len(nodes) - 1 and nodes[ind + 1] != None:
                            if term.lower() == aspect_list[0].lower() and \
                                nodes[ind + 1].lower() == aspect_list[1].lower():
                                tree.get(ind).trueLabel = 1
                                tree.aspect = 1
                                for i in range(len(aspect_list) - 1):
                                    if ind + i + 1 < len(nodes) and nodes[ind + i + 1] != None:
                                        if nodes[ind + i + 1].lower() == aspect_list[i + 1].lower():
                                            tree.get(ind + i + 1).trueLabel = 2
                                break
                #aspect is a single word
                else:
                    for ind, term in enumerate(nodes):
                        if term != None:
                            if term.lower() == target.lower() and tree.get(ind).trueLabel == 0:
                                tree.get(ind).trueLabel = 1
                                tree.aspect = 1

        for term in nodes:
            if term != None:
                ind = nodes.index(term)
                tree.get(ind).word = term.lower()
 
                    
        # add dependency edges between nodes
        for rel, deps in plist:
            par_ind, par_word = deps[0]
            kid_ind, kid_word = deps[1]
            tree.add_edge(par_ind, kid_ind, rel) 
        
        # source domain = 1, target domain = 0
        tree.domain = 1
        tree_source.append(tree) 
        # store vocabulary
        for node in tree.get_nodes():
            if node.word.lower() not in vocab:
                vocab.append(node.word.lower())                
            node.ind = vocab.index(node.word.lower())            
            for ind, rel in node.kids:
                if rel not in rel_list:
                    rel_list.append(rel)

        plist = []


# process target tree files
for line in data_target:
    # construct dependency tree structure for each sentence
    if line.strip():
        rel_split = line.split('(')
        rel = rel_split[0]
        deps = rel_split[1][:-1]
        deps = deps.replace(')','')
        if len(rel_split) != 2:
            print 'error ', rel_split
            sys.exit(0)

        else:
            dep_split = deps.split(',')
            
        if len(dep_split) > 2:
            fixed = []
            half = ''
            for piece in dep_split:
                piece = piece.strip()
                if '-' not in piece:
                    half += piece

                else:
                    fixed.append(half + piece)
                    half = ''

            dep_split = fixed

        final_deps = []
        for dep in dep_split:
            words = dep.split('-')
            word = words[0]
            ind = int(words[len(words) - 1])

            if len(words) > 2:
                word = '-'.join([w for w in words[:-1]])

            final_deps.append( (ind, word.strip()) )
        # store relations and word with indices    
        plist.append((rel,final_deps))

    # end of a sentence, store all the dependencies into tree format
    else:
        edges=[]
        max_ind = -1
        for rel, deps in plist:
            for ind, word in deps:
                if ind > max_ind:
                    max_ind = ind
        # load words into nodes, then make a dependency tree
        nodes = [None for i in range(0, max_ind + 1)]
        for rel, deps in plist:
            for ind, word in deps:
                nodes[ind] = word
        tree = dtree(nodes)

        # label the words in each sentence with ground-truth labels
        aspect_term = label_file_target.readline().strip()
        labeled_sent = label_sentence_target.readline().strip()    
            
        #facilitate bio notation
        aspect_BIO = {}
        # if opinion words exist
        if '##' in labeled_sent:
                opinions = labeled_sent.split('##')[1].strip()
                opinions = opinions.split(',')                
                for opinion in opinions:
                    if opinion != '':
                        op_list = opinion.split()[:-1]
                        pol = opinion.split()[-1]
                        if len(op_list) > 1:
                            for ind, term in enumerate(nodes):
                                if term != None:
                                    if term.lower() == op_list[0].lower() and ind < len(nodes) - 1 and nodes[ind + 1] != None\
                                      and nodes[ind + 1].lower() == op_list[1].lower():
                                        tree.get(ind).trueLabel = 3
                                        for i in range(len(op_list) - 1):
                                            if nodes[ind + i + 1].lower() == op_list[i + 1].lower():
                                                tree.get(ind + i + 1).trueLabel = 4
                                            
                        elif len(op_list) == 1:
                            for ind, term in enumerate(nodes):
                                if term != None:
                                    if term.lower() == op_list[0].lower() and tree.get(ind).trueLabel == 0:
                                        tree.get(ind).trueLabel = 3
        # if aspect words exist
        if aspect_term != 'NIL':
            aspects = aspect_term.split(',')
            for aspect in aspects:
                aspect = aspect.strip()
                target = aspect.split(':')[0]
                #aspect is a phrase
                if ' ' in target:
                    aspect_list = target.split()
                    for ind, term in enumerate(nodes):
                        if term != None and ind < len(nodes) - 1 and nodes[ind + 1] != None:
                            if term.lower() == aspect_list[0].lower() and \
                                nodes[ind + 1].lower() == aspect_list[1].lower():
                                tree.get(ind).trueLabel = 1
                                tree.aspect = 1
                                for i in range(len(aspect_list) - 1):
                                    if ind + i + 1 < len(nodes) and nodes[ind + i + 1] != None:
                                        if nodes[ind + i + 1].lower() == aspect_list[i + 1].lower():
                                            tree.get(ind + i + 1).trueLabel = 2
                                break
                #aspect is a single word
                else:
                    for ind, term in enumerate(nodes):
                        if term != None:
                            if term.lower() == target.lower() and tree.get(ind).trueLabel == 0:
                                tree.get(ind).trueLabel = 1
                                tree.aspect = 1

        for term in nodes:
            if term != None:
                ind = nodes.index(term)
                tree.get(ind).word = term.lower()
 
                    
        # add dependency edges between nodes
        for rel, deps in plist:
            par_ind, par_word = deps[0]
            kid_ind, kid_word = deps[1]
            tree.add_edge(par_ind, kid_ind, rel) 
        
        # source domain = 1, target domain = 0
        tree.domain = 0
        tree_target.append(tree) 
        # store vocabulary
        for node in tree.get_nodes():
            if node.word.lower() not in vocab:
                vocab.append(node.word.lower())                
            node.ind = vocab.index(node.word.lower())            
            for ind, rel in node.kids:
                if rel not in rel_list:
                    rel_list.append(rel)

        plist = []


# shuffle the source and target domain 3 times and split into 4 groups        
ind_source = range(len(tree_source))
ind_target = range(len(tree_target))
t_source = list(zip(ind_source, tree_source))
t_target = list(zip(ind_target, tree_target))

random.seed(1)
random.shuffle(t_source)
random.shuffle(t_target)
is_1, ts_1 = zip(*t_source)
it_1, tt_1 = zip(*t_target)
random.shuffle(t_source)
random.shuffle(t_target)
is_2, ts_2 = zip(*t_source)
it_2, tt_2 = zip(*t_target)
random.shuffle(t_source)
random.shuffle(t_target)
is_3, ts_3 = zip(*t_source)
it_3, tt_3 = zip(*t_target)

s1_test, s1_train = ts_1[:len(t_source)/4], ts_1[len(t_source)/4:]
s2_test, s2_train = ts_2[:len(t_source)/4], ts_2[len(t_source)/4:]
s3_test, s3_train = ts_3[:len(t_source)/4], ts_3[len(t_source)/4:]

t1_test, t1_train = tt_1[:len(t_target)/4], tt_1[len(t_target)/4:]
t2_test, t2_train = tt_2[:len(t_target)/4], tt_2[len(t_target)/4:]
t3_test, t3_train = tt_3[:len(t_target)/4], tt_3[len(t_target)/4:]

train1 = s1_train + t1_train
testt1 = t1_test
tests1 = s1_test
train2 = s2_train + t2_train
testt2 = t2_test
tests2 = s2_test
train3 = s3_train + t3_train
testt3 = t3_test
tests3 = s3_test


print 'rels: ', len(rel_list)
print 'vocab: ', len(vocab)

cPickle.dump((vocab, rel_list, train1, train2, train3, testt1, tests1, testt2, tests2, testt3, tests3, \
            is_1, is_2, is_3, it_1, it_2, it_3), \
            open("data_semEval/final_input_lapdev_split4", "wb"))

cPickle.dump((is_1, is_2, is_3, it_1, it_2, it_3), open('shuffle4_lapdev1.idx', 'wb'))


