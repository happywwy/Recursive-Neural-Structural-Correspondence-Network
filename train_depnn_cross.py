
"""
pre-train the recursive neural network first
"""

import numpy as np
from util.gen_util import *
from util.math_util import *
from util.dtree_util import *
from rnn.rmsprop import Rmsprop
import rnn.propagation_rnn as prop
import cPickle, time, argparse
import random
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
    recall = float(correct) / (relevant + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    print relevant
    
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
    recall = float(correct) / (relevant + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    print relevant
    
    return precision, recall, f1
    

def evaluate(model, out, epoch, test_trees, rel_dict, Wv, Wh, Wc, b, b_c, We, vocab, rel_list, d, c, mixed = False):

    params = (rel_dict, Wv, Wh, Wc, b, b_c, We)    
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
        
      
    true_list = []
    predict_list = []
    rel_all = 0
    rel_match_all = 0
    
    count = 0
    
    for tree in test_trees:
        true = []
        predict = []
        for node in tree.get_nodes():
            if node.word.lower() in vocab:
                node.vec = We[:, node.ind].reshape( (d, 1) )
        # evaluate relation prediction
        match_rel, total_rel = prop.forward_prop(model, params, rel_list, tree, d, c, labels=False)
        rel_all += total_rel
        rel_match_all += match_rel
        
        for node in tree.get_nodes()[1:]:
            max = 0
            predict_label = node.predict_label
            for entry in predict_label:
                if entry > max:
                    max = entry
            node.prediction = np.nonzero(predict_label==(max))[0][0]
            #add prediction to predict list
            predict.append(node.prediction)
            true.append(node.trueLabel)
            
            if node.prediction == 1:
                count += 1

        true_list.append([str(y) for y in true])
        predict_list.append([str(y) for y in predict])
        
    precision_aspect, recall_aspect, f1_aspect = score_aspect(true_list, predict_list)
    precision_op, recall_op, f1_op = score_opinion(true_list, predict_list)
    
    print "relation accuracy: \n", float(rel_match_all) / rel_all    
    print "precision_aspect: \n", precision_aspect
    print "recall_aspect: \n", recall_aspect
    print "f1_aspect: \n", f1_aspect
    print "precision_opinion: \n", precision_op
    print "recall_opinion: \n", recall_op
    print "f1_opinion: \n", f1_op
    print "Predicted number of aspects:", count
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

#computes parameter updates with regularization
def par_objective(rel_out, model, data, rel_dict, Wv, Wh, Wc, b, b_c, L, d, c, len_voc, rel_list, lambdas):

    #non-data params
    params = (rel_dict, Wv, Wh, Wc, b, b_c, L)
    oparams = [params, d, c, len_voc, rel_list]

    param_data = []
    param_data.append(oparams)
    param_data.append(data)
    
    #gradient and error  
    result = objective_and_grad(rel_out, model, param_data)
    [total_err, grads, all_nodes] = result

    # add L2 regularization
    [lambda_W, lambda_L, lambda_C] = lambdas

    reg_cost = 0.0
    #regularization for relation matrices
    for key in rel_list:
        reg_cost += 0.5 * lambda_W * sum(rel_dict[key] ** 2)
        grads[0][key] = grads[0][key] / all_nodes
        grads[0][key] += lambda_W * rel_dict[key]
    
    #regularization for transformation matrix Wv
    reg_cost += 0.5 * lambda_W * sum(Wv ** 2)
    grads[1] = grads[1] / all_nodes
    grads[1] += lambda_W * Wv
    
    reg_cost += 0.5 * lambda_W * sum(Wh ** 2)
    grads[2] = grads[2] / all_nodes
    grads[2] += lambda_W * Wh
    
    #regularization for classification matrix Wc
    reg_cost += 0.5 * lambda_C * sum(Wc ** 2)
    grads[3] = grads[3] / all_nodes
    grads[3] += lambda_C * Wc

    #regularization for bias b
    grads[4] = grads[4] / all_nodes
    
    #regularization for bias b_c
    grads[5] = grads[5] / all_nodes

    reg_cost += 0.5 * lambda_L * sum(L ** 2)
    
    #regularization for word embedding matrix
    grads[6] = grads[6] / all_nodes
    grads[6] += lambda_L * L

    cost = total_err / all_nodes + reg_cost

    return cost, grads


# this function computes the objective / grad for each minibatch
def objective_and_grad(rel_out, model, par_data):

    params, d, c, len_voc, rel_list = par_data[0]
    data = par_data[1]
    
    # returns list of initialized zero gradients which backprop modifies
    grads = init_dtrnn_scl_auto_grads(rel_list, d, c, len_voc)
    (rel_dict, Wv, Wh, Wc, b, b_c, L) = params

    error_sum = 0.0
    tree_size_all = 0
    rel_all = 0
    rel_match_all = 0

    # compute error and gradient for each tree in minibatch
    # also keep track of total number of nodes in minibatch
    for index, tree in enumerate(data):
        if len(tree.get_nodes()) <= 2:
            continue

        nodes = tree.get_nodes()
        for node in nodes:
            node.vec = L[:, node.ind].reshape( (d, 1) ) 
        # if the tree is in source domain, using labeled data to backpropagate
        if tree.domain:
            match_rel, total_rel = prop.forward_prop(model, params, rel_list, tree, d, c, True, rel_out)
            prop.backprop_s(model, params[:-1], rel_list, tree, d, c, len_voc, grads)
            error_sum += tree.error()
        # if the tree is in target domain, only use auxiliary task to backpropagate
        else:
            match_rel, total_rel = prop.forward_prop(model, params, rel_list, tree, d, c, False, rel_out)
            prop.backprop_t(model, params[:-1], rel_list, tree, d, c, len_voc, grads)
            
        error_sum += tree.error_aux            
        tree_size_all += len(nodes)
        
        rel_all += total_rel
        rel_match_all += match_rel
        
    #print float(rel_match_all) / rel_all
    return (error_sum, grads, tree_size_all)
    

# train and save model
if __name__ == '__main__':
    
    seed_list = [12]
    for seed_i in seed_list:

        # command line arguments
        parser = argparse.ArgumentParser(description='A dependency tree-based recursive neural network')
        parser.add_argument('-data', help='location of dataset', default='util/data_semEval/final_input_lapdev_split4')
        parser.add_argument('-We', help='location of word embeddings', default='util/data_semEval/word_embeddings100_lapdev_norm')
        parser.add_argument('-d', help='word embedding dimension', type=int, default=100)
        
        # no of classes
        parser.add_argument('-c', help='number of classes', type=int, default=5)
        parser.add_argument('-lW', '--lambda_W', help='regularization weight for composition matrices', \
                            type=float, default=0.0001)
        parser.add_argument('-lWe', '--lambda_We', help='regularization weight for word embeddings', \
                            type=float, default=0.0001)
        # regularization for classification matrix
        parser.add_argument('-lWc', '--lambda_Wc', help='regularization weight for classification matrix', \
                            type=float, default=0.0001)                    
                        
        parser.add_argument('-b', '--batch_size', help='rmsprop minibatch size', type=int,\
                            default=30)
        parser.add_argument('-ep', '--num_epochs', help='number of training epochs', type=int, default=3)
        parser.add_argument('-agr', '--adagrad_reset', help='reset sum of squared gradients after this many\
                             epochs', type=int, default=30)

        parser.add_argument('-o', '--output', help='desired location of output model', \
                             default='models/trainingLapDev100_params_rnn_4split')
                             
        parser.add_argument('-op', help='use mixed word vector or not', default = False)
        parser.add_argument('-len', help='training vector length', default = 50)
    
        args = vars(parser.parse_args())
        out = open('result_rnn_lapTodev_4split.txt', 'a')
        
    
        ## load data
        (vocab, rel_list, train1, train2, train3, testt1, tests1, testt2, tests2, testt3, tests3, \
            is_1, is_2, is_3, it_1, it_2, it_3) = cPickle.load(open(args['data'], 'rb'))
        # target training data
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
        # target test data for inductive testing
        test_list = [testt1, testt2, testt3]
        rel_list.remove('root')

        # regularization lambdas
        lambdas = [args['lambda_W'], args['lambda_We'], args['lambda_Wc']]
        # iterate over all different split
        for split, train_trees, test_trees, in_domain in zip(range(1,4), train_list, test_list, transduct):
            # word embedding matrix
            orig_We = cPickle.load(open(args['We'], 'rb'))
            # output log and parameter file destinations
            param_file = args['output'] + str(split)
            # "training_log"
            log_file = param_file.split('_')[0] + '_log'
            # initialize autoencoder model, 20 relation groups
            model = Groups(input_dim=args['d'], emb_dim=args['d'], n_groups=20, nc=len(rel_list))
            model.build(loss1="crossentropy", loss2="mse", optimizer="rmsprop", lr=0.01, rho=0.9, epsilon=1e-6)

        
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
        
            # pop bad trees, higher indices first
            # print 'removed ', len(bad_trees)
            for ind in bad_trees[::-1]:
                train_trees = np.delete(train_trees, ind)
        
    
            # generate params
            params = gen_dtrnn_scl_auto_params(args['d'], args['c'], len(rel_list), rel_list)        
            # add Word embedding matrix to params
            params += (orig_We, )
            # r is 1-D param vector, roll all the params into a vector
            r = roll_scl_auto_params(params, rel_list)
            dim = r.shape[0]
            print 'parameter vector dimensionality:', dim
            log = open(log_file, 'w')
            paramfile = open( param_file, 'wb')
        
            # minibatch rmsprop training
            rm = Rmsprop(r.shape)
            (rel_dict, Wv, Wh, Wc, b, b_c, We) = params
    
            for tdata in [train_trees]:
                min_error = float('inf')
                for epoch in range(0, args['num_epochs']):
                    rel_out = open('rel_group.txt', 'w')
                    lstring = ''
                    
                    random.seed(seed_i)
                    # shuffle and create mini-batches
                    random.shuffle(tdata)
                    batches = [tdata[x : x + args['batch_size']] for x in xrange(0, len(tdata), 
                               args['batch_size'])]
        
                    epoch_error = 0.0
                    for batch_ind, batch in enumerate(batches):
                        now = time.time()
                        # return cost, grad  
                        if args['op']:
                            err, grads = par_objective(rel_out, model, batch, rel_dict, Wv, Wh, Wc, b, b_c, We, args['d'] + args['len'], \
                                                       args['c'], len(vocab), rel_list, lambdas)
                        else:
                            err, grads = par_objective(rel_out, model, batch, rel_dict, Wv, Wh, Wc, b, b_c, We, args['d'], \
                                                       args['c'], len(vocab), rel_list, lambdas)
                        # roll gradients into a 1-D vector for rmsprop rescale                       
                        grad = roll_scl_auto_params(grads, rel_list)                          
                        update = rm.rescale_update(grad)
                        updates = unroll_scl_auto_params(update, args['d'], args['c'], len(rel_list), len(vocab), rel_list)
                        for rel in rel_list:
                            rel_dict[rel] -= updates[0][rel]
                        Wv -= updates[1]
                        Wh -= updates[2]
                        Wc -= updates[3]
                        b -= updates[4]
                        b_c -= updates[5]
                        We -= updates[6]
    
                        lstring = 'epoch: ' + str(epoch) + ' batch_ind: ' + str(batch_ind) + \
                                ' error, ' + str(err) + ' time = '+ str(time.time()-now) + ' sec'
                        print lstring
                        log.write(lstring + '\n')
                        log.flush()
        
                        epoch_error += err
                        
                    rel_out.close()
                    print 'evaluation on in-sample data: \n'
                    out.write('evaluation on in-sample data: \n')
                    evaluate(model, out, epoch, in_domain, rel_dict, Wv, Wh, Wc, b, b_c, We, vocab, rel_list, d, c, mixed = False)
                    print 'evaluation on out-sample data: \n'
                    out.write('evaluation on out-sample data: \n')
                    evaluate(model, out, epoch, test_trees, rel_dict, Wv, Wh, Wc, b, b_c, We, vocab, rel_list, d, c, mixed = False)
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
                        params = (rel_dict, Wv, Wh, Wc, b, b_c, We)

            
            cPickle.dump( ( params, vocab, rel_list), paramfile)            
            log.close()
            paramfile.close()
            # save params from autoencoder
            auto_file = open('discrete_autoencoder_params_lapdev_'+str(split), 'wb')
            model.save(auto_file)
            auto_file.close()
            
    

