# -*- coding: utf-8 -*-
import numpy as np
import util.gen_util as gen_util
from rnn.rmsprop_joint import Rmsprop
import rnn.propagation_joint as prop
import cPickle, time, argparse

import random
import gru_batch
from rnn.discrete_auto import Groups

#f1 score
def score_aspect(true_list, predict_list):
    
    correct = 0
    predicted = 0
    relevant = 0
    
    i=0
    j=0
    pairs = []
    while i < len(true_list):
        true_seq = true_list[i]
        predict = predict_list[i]
        
        for num in range(len(true_seq)):
            if true_seq[num] == '1':
                if num < len(true_seq) - 1:
                    if true_seq[num + 1] != '2':
                        if predict[num] == '1' and predict[num + 1] != '2':
                            correct += 1
                            relevant += 1
                        else:
                            relevant += 1
                    
                    else:
                        if predict[num] == '1':
                            for j in range(num + 1, len(true_seq)):
                                if true_seq[j] == '2':
                                    if predict[j] == '2' and j < len(predict) - 1:
                                        continue
                                    elif predict[j] == '2' and j == len(predict) - 1:
                                        correct += 1
                                        relevant += 1
                                        
                                    else:
                                        relevant += 1
                                        break
                                    
                                else:
                                    if predict[j] != '2':
                                        correct += 1
                                        relevant += 1
                                        break
    
                                
                        else:
                            relevant += 1
                            
                else:
                    if predict[num] == '1':
                        correct += 1
                        relevant += 1
                    else:
                        relevant += 1
                        
                            
        for num in range(len(predict)):
            if predict[num] == '1':
                predicted += 1
      
        i += 1
                
    precision = float(correct) / (predicted + 1e-6)
    recall = float(correct) / relevant
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return precision, recall, f1


def score_opinion(true_list, predict_list):
    
    correct = 0
    predicted = 0
    relevant = 0
    
    i=0
    j=0
    pairs = []
    while i < len(true_list):
        true_seq = true_list[i]
        predict = predict_list[i]
        
        for num in range(len(true_seq)):
            if true_seq[num] == '3':
                if num < len(true_seq) - 1:
                    if true_seq[num + 1] != '4':
                        if predict[num] == '3' and predict[num + 1] != '4':
                            correct += 1
                            relevant += 1
                        else:
                            relevant += 1
                    
                    else:
                        if predict[num] == '3':
                            for j in range(num + 1, len(true_seq)):
                                if true_seq[j] == '4':
                                    if predict[j] == '4' and j < len(predict) - 1:
                                        continue
                                    elif predict[j] == '4' and j == len(predict) - 1:
                                        correct += 1
                                        relevant += 1
                                        
                                    else:
                                        relevant += 1
                                        break
                                    
                                else:
                                    if predict[j] != '4':
                                        correct += 1
                                        relevant += 1
                                        break
    
                                
                        else:
                            relevant += 1
                            
                else:
                    if predict[num] == '3':
                        correct += 1
                        relevant += 1
                    else:
                        relevant += 1
                        
                            
        for num in range(len(predict)):
            if predict[num] == '3':
                predicted += 1

        i += 1
        
    precision = float(correct) / (predicted + 1e-6)
    recall = float(correct) / relevant
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return precision, recall, f1


    
def minibatch(l, bs):
    '''
    l :: list of word idxs
    return a list of minibatches of indexes
    which size is equal to bs
    border cases are treated as follow:
    eg: [0,1,2,3] and bs = 3
    will output:
    [[0],[0,1],[0,1,2],[1,2,3]]
    '''
    out  = [l[:i] for i in xrange(1, min(bs,len(l)+1) )]
    out += [l[i-bs:i] for i in xrange(bs,len(l)+1) ]
    assert len(l) == len(out)
    return out

def contextwin(l, win, seq_size):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >=1
    l = list(l)

    lpadded = win/2 * [seq_size - 2] + l + win/2 * [seq_size - 2]
    out = [ lpadded[i:i+win] for i in range(len(l)) ]

    assert len(out) == len(l)
    return out
    
    
def evaluate(model, s, out, epoch, aux, test_trees, params, rel_list, vocab, lstm_model, d, c, mixed = False):
    
    [rel_dict, Wv, Wh, b, We] = params
    
    bad_trees = []
    for ind, tree in enumerate(test_trees):
        if len(tree.nodes) == 0:
            bad_trees.append(ind)
            continue
        elif tree.get(0).is_word == 0:
            # print tree.get_words()
            bad_trees.append(ind)
            continue
        elif len(tree.get_nodes()) <= 2:
            bad_trees.append(ind)
            continue

    # print 'removed', len(bad_trees)
    for ind in bad_trees[::-1]:
        #test_trees.pop(ind)
        test_trees = np.delete(test_trees, ind)
        
    true = []
    predict = []    
    true_list = []
    predict_list = []
    
    for tree in test_trees:
        nodes = tree.get_nodes()
        
        for node in nodes:
            # retrieve word embeddings
            if node.word.lower() in vocab:
                node.vec = We[:, node.ind].reshape(d, 1)
                
        prop.forward_prop_auto(model, (rel_dict, Wv, Wh, b, We), rel_list, tree, d, c, labels=False)
        
        # calculate gru on top of the hidden vectors
        # initialize the hidden vectors as the input to GRU, batch
        h_input = np.zeros((1, len(tree.get_nodes()) + 1, d))
        mask = [len(tree.nodes) - 1]
        y_label = []
        index2word = []
        
        index = 0
        for ind, node in enumerate(tree.nodes):
            # non-root
            if ind != 0:
                if tree.get(ind).is_word == 0:
                    y_label.append(0)
                    index2word.append(len(tree.get_nodes()))
                    
                else:
                    y_label.append(node.trueLabel)
                    index2word.append(index)
                
                    for i in range(d):
                        h_input[0][index][i] = node.p[i]
                    index += 1
        
        # add embeddings for padding and punctuation
        for i in range(d):
            h_input[0][len(tree.get_nodes()) - 1][i] = aux['padding'][i]
            h_input[0][len(tree.get_nodes())][i] = aux['punkt'][i]
                    
        #convert to gru input format
        idxs = np.asarray(contextwin(index2word, s['win'], h_input.shape[1])).reshape((1, len(tree.nodes)-1, s['win']))
        pred_y = lstm_model.predict(idxs, h_input, mask)[0]
        
        true_list.append([str(y) for y in y_label])
        predict_list.append([str(y) for y in pred_y])
        
            
    precision_aspect, recall_aspect, f1_aspect = score_aspect(true_list, predict_list)
    precision_op, recall_op, f1_op = score_opinion(true_list, predict_list)
    
    print "precision_aspect: \n", precision_aspect
    print "recall_aspect: \n", recall_aspect
    print "f1_aspect: \n", f1_aspect
    print "precision_opinion: \n", precision_op
    print "recall_opinion: \n", recall_op
    print "f1_opinion: \n", f1_op
    out.write(str(epoch))
    out.write('\n')
    out.write("aspect_precision: ")
    out.write(str(precision_aspect))
    out.write("aspect_recall: ")
    out.write(str(recall_aspect))
    out.write("aspect_f1: ")
    out.write(str(f1_aspect))
    out.write('\n')
    out.write("opinion_precision: ")
    out.write(str(precision_op))
    out.write("opinion_recall: ")
    out.write(str(recall_op))
    out.write("opinion_f1: ")
    out.write(str(f1_op))
    out.write('\n')

# splits the training data into minibatches
def par_objective(model, s, lstm_model, aux, data, rel_dict, Wv, Wh, b, L, d, c, len_voc, rel_list, lambdas, train_size):

    grads = gen_util.init_joint_auto_grads(rel_list, d, c, len_voc)
    
    error = 0.0
    tree_size = 0
    source_num = 0
    
    max_size = 0
    for tree in data:
        if len(tree.nodes) > max_size:
            max_size = len(tree.nodes)
        if tree.domain:
            source_num += 1
    
    # obtain hidden vectors from recursive neural network
    # compute batch input representations for GRU
    h_input = np.zeros((source_num, max_size + 5, d))
    y_label = np.zeros((source_num, max_size))
    index2word = np.zeros((source_num, max_size))
    context_words = []
    mask = []
    
    i = 0
    for tree in data:
        nodes = tree.get_nodes()
        for node in nodes:
            node.vec = L[:, node.ind].reshape( (d, 1) )

        prop.forward_prop_auto(model, [rel_dict, Wv, Wh, b, L], rel_list, tree, d, c)
    
        tree_size += len(nodes)
        # if the tree is from source domain, use word labels
        if tree.domain:# and hasattr(tree, 'aspect'):
            word_index = 0
            for ind, node in enumerate(tree.nodes[1:]):
                if tree.get(ind+1).is_word == 0:
                    y_label[i][ind] = 0
                    index2word[i][ind] = len(tree.get_nodes())
                    
                else:
                    y_label[i][ind] = node.trueLabel
                    index2word[i][ind] = word_index
                    
                    for k in range(d):
                        h_input[i][word_index][k] = node.p[k]
                    
                    word_index += 1
            # pass hidden vectors to GRU
            for n in range(d):
                h_input[i][len(tree.get_nodes()) - 1][n] = aux['padding'][n]
                h_input[i][len(tree.get_nodes())][n] = aux['punkt'][n]                
            
            # convert to GRU input
            context_words.append(contextwin(index2word[i], s['win'], h_input.shape[0]))
            
            i += 1
            mask.append(len(tree.nodes) - 1)
        
        # if tree is from target domain, directly compute gradients
        else:
            prop.backprop_t_auto([rel_dict, Wv, Wh, b], rel_list, tree, d, c, len_voc, grads)
        # auxiliary loss
        error += tree.error_aux
    
    if source_num > 0:       
        error += lstm_model.train(context_words,h_input,y_label,s['lr'],mask)    
        grad_emb = lstm_model.grad(context_words,h_input,y_label,mask)
    
    i = 0
    for tree in data:
        if tree.domain:
            # update padding vectors
            aux['padding'] -= s['lr'] / source_num * grad_emb[i][len(tree.get_nodes()) - 1, :]
            aux['punkt'] -= s['lr'] / source_num * grad_emb[i][len(tree.get_nodes()), :]
    
            # update hidden vectors
            for ind, node in enumerate(tree.get_nodes()[1:]):
                node.grad_emb = grad_emb[i][ind].reshape(d,1)
            # pass gradient from hidden vectors to the tree
            prop.backprop_s_auto([rel_dict, Wv, Wh, b], rel_list, tree, d, c, len_voc, grads)
            i += 1
    
    
    [lambda_W, lambda_L] = lambdas
    for key in rel_list:
        grads[0][key] = grads[0][key] / tree_size
        grads[0][key] += lambda_W * rel_dict[key]

    grads[1] = grads[1] / tree_size
    grads[1] += lambda_W * Wv
    grads[2] = grads[2] / tree_size
    grads[2] += lambda_W * Wh
    grads[3] = grads[3] / tree_size
    grads[4] = grads[4] / tree_size
    grads[4] += lambda_L * L

    cost = error
    return cost, grads, aux



# train qanta and save model
if __name__ == '__main__':
    

    # command line arguments
    parser = argparse.ArgumentParser(description='A joint model for cross-domain extraction')
    parser.add_argument('-data', help='location of dataset', default='util/data_semEval/final_input_lapdev_split4')
    #add model parameter
    parser.add_argument('-pretrain_model', help='location of pretrained model', default='models/trainingLapDev100_params_rnn_4split')
    parser.add_argument('-d', help='word embedding dimension', type=int, default=100)
    
    # no of classes
    parser.add_argument('-c', help='number of classes', type=int, default=5)
    parser.add_argument('-lW', '--lambda_W', help='regularization weight for composition matrices', \
                        type=float, default=0.0001)
    parser.add_argument('-lWe', '--lambda_We', help='regularization weight for word embeddings', \
                        type=float, default=0.0001)
                    
    parser.add_argument('-b', '--batch_size', help='rmsprop minibatch size', type=int,\
                        default=30)
    parser.add_argument('-ep', '--num_epochs', help='number of training epochs', \
                        type=int, default=10)

    parser.add_argument('-o', '--output', help='desired location of output model', \
                         default='final_model/params_joint')
                         
    parser.add_argument('-op', help='use mixed word vector or not', default = False)
    parser.add_argument('-len', help='training vector length', default = 50)

    args = vars(parser.parse_args())
    outcome = open('outcomes_joint_lapTodev_100_4split.txt', 'a')
    
    
 
    (vocab, rel_list, train1, train2, train3, testt1, tests1, testt2, tests2, testt3, tests3, \
            is_1, is_2, is_3, it_1, it_2, it_3) = cPickle.load(open(args['data'], 'rb'))
    lambdas = [args['lambda_W'], args['lambda_We']]
    # target domain training data
    train1target = [tree for tree in train1 if not tree.domain]
    train2target = [tree for tree in train2 if not tree.domain]
    train3target = [tree for tree in train3 if not tree.domain]
    transduct = [train1target, train2target, train3target]
    '''  
    # use this when device is the source domain
    train1 = [tree for tree in train1 if (not tree.domain and hasattr(tree, 'aspect')) or tree.domain]
    train2 = [tree for tree in train2 if (not tree.domain and hasattr(tree, 'aspect')) or tree.domain]
    train3 = [tree for tree in train3 if (not tree.domain and hasattr(tree, 'aspect')) or tree.domain]
    '''
    train_list = [train1, train2, train3]
    # target domain test data
    test_list = [testt1, testt2, testt3]
    
    for rate in [0.01]:
        # iterate over different split
        for split, train_trees, test_trees, in_domain in zip(range(1,4), train_list, test_list, transduct):
            #build gru model
            s = {
        		'lr':rate,
        		'win':3, # number of words in the context window
        		'nhidden':50, # number of hidden units
        		'emb_dimension':100, # dimension of word embedding
        		}
          
            lstm_model = gru_batch.model(nh=s['nhidden'],
                                    nc=5,
                                    ne=100,
                                    de=s['emb_dimension'],
                                    cs=s['win'],
                                    decay=0.9)
            # retrieve autoencoder model
            autofile = open('discrete_autoencoder_params_lapdev_'+str(split), 'rb')
            autoparams = cPickle.load(autofile)
            autofile.close()
            # construct autoencoder model with retrieved params
            model = Groups(input_dim=args['d'], emb_dim=args['d'], n_groups=20, nc=len(rel_list), params=autoparams)
            model.build(loss1="crossentropy", loss2="mse", optimizer="rmsprop", lr=0.001, rho=0.9, epsilon=1e-6)

            #import pre-trained model parameters
            params, vocab, rel_list = cPickle.load(open(args['pretrain_model'] + str(split), 'rb'))
            (rel_dict, Wv, Wh, Wc, b, b_c, We) = params
        
            # output log and parameter file destinations
            # "training_param"
            param_file = args['output']
            # "training_log"
            log_file = param_file.split('_')[0] + '_log'
        
            print 'number of training sentences:', len(train_trees)
            print 'number of dependency relations:', len(rel_list)
            print 'number of classes:', args['c']
            
            c = args['c']
            d = args['d']
        
            ## remove incorrectly parsed sentences from data
            # print 'removing bad trees train...'
            bad_trees = []
            for ind, tree in enumerate(train_trees):
                
                #add condition when the tree is empty
                if tree.get_nodes() == []:
                    bad_trees.append(ind)
        
                elif tree.get(0).is_word == 0:
                    print tree.get_words(), ind
                    bad_trees.append(ind)
                elif len(tree.get_nodes()) <= 2:
                    bad_trees.append(ind)
        
            # pop bad trees, higher indices first
            # print 'removed ', len(bad_trees)
            for ind in bad_trees[::-1]:
                #train_trees.pop(ind)
                train_trees = np.delete(train_trees, ind)

            train_size = len(train_trees)
            # r is 1-D param vector
            r = gen_util.roll_params_auto_joint((rel_dict, Wv, Wh, b, We), rel_list)
        
            dim = r.shape[0]
            print 'parameter vector dimensionality:', dim
            log = open(log_file, 'w')            
        
            # minibatch adagrad training
            rm = Rmsprop(r.shape)
            aux = {'padding':np.random.uniform(-0.2, 0.2, (d,)), 'punkt':np.random.uniform(-0.2, 0.2, (d,))}
            for tdata in [train_trees]:
        
                min_error = float('inf')
                for epoch in range(0, args['num_epochs']):
                    
                    param_file = args['output'] + str(epoch)
                    paramfile = open( param_file, 'wb')
                    lstring = ''
        
                    # create mini-batches
                    random.seed(12)
                    random.shuffle(tdata)
                    batches = [tdata[x : x + args['batch_size']] for x in xrange(0, len(tdata), 
                               args['batch_size'])]
        
                    epoch_error = 0.0
                    
                    for batch_ind, batch in enumerate(batches):
                        now = time.time()
                        # return loss and gradients
                        err, grad, aux = par_objective(model, s, lstm_model, aux, batch, rel_dict, Wv, Wh, b, We, \
                          args['d'], args['c'], len(vocab), rel_list, lambdas, train_size)
                        
                        grad_vec = gen_util.roll_params_auto_joint(grad, rel_list)
                        update = rm.rescale_update(grad_vec)
                        gradient = gen_util.unroll_params_auto_joint(update, d, c, len(vocab), rel_list)
                        
                        for rel in rel_list:
                            rel_dict[rel] -= gradient[0][rel]
                            
                        Wv -= gradient[1]
                        Wh -= gradient[2]
                        b -= gradient[3]
                        We -= gradient[4]
                                
                        lstring = 'epoch: ' + str(epoch) + ' batch_ind: ' + str(batch_ind) + \
                                ' error, ' + str(err) + ' time = '+ str(time.time()-now) + ' sec'
                        print lstring
                        log.write(lstring + '\n')
                        log.flush()
        
                        epoch_error += err
                                                
                    print 'evaluation on in-sample data: \n'
                    outcome.write('evaluation on in-sample data: \n')
                    evaluate(model, s, outcome, epoch, aux, in_domain, [rel_dict, Wv, Wh, b, We], rel_list, vocab, lstm_model, d, c)
                    print 'evaluation on out-sample data: \n'
                    outcome.write('evaluation on out-sample data: \n')
                    evaluate(model, s, outcome, epoch, aux, test_trees, [rel_dict, Wv, Wh, b, We], rel_list, vocab, lstm_model, d, c)
                        
                    # done with epoch
                    print 'done with epoch ', epoch, ' epoch error = ', epoch_error, ' min error = ', min_error
                    lstring = 'done with epoch ' + str(epoch) + ' epoch error = ' + str(epoch_error) \
                             + ' min error = ' + str(min_error) + '\n\n'
                    log.write(lstring)
                    log.flush()
                    # save parameters if the current model is better than previous best model
                    if epoch_error < min_error:
                        min_error = epoch_error
                        print 'saving model...'
        
        
            log.close()
    

    
    



