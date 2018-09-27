import numpy as np
from util.math_util import *
import random

#   forward and backward propagation. the labels argument indicates whether
#   you want to compute errors and deltas at each node or not. 
#   for computations in the joint model

#define softmax function
def softmax(v):
    v = np.array(v)
    max_v = np.amax(v)
    e = np.exp(v - max_v)
    dist = e / np.sum(e)

    return dist

    
def der_tanh(x):
    return 1-np.tanh(x)**2



# forward computation for RNN within the joint model
# The difference with propagation_rnn is that no classification parameters are required
def forward_prop_auto(model, params, rel_list, tree, d, c, labels=True):

    tree.reset_finished()
    to_do = tree.get_nodes()
    (rel_dict, Wv, Wh, b, We) = params

    X_c = []
    Xc_ind = []
    Y = []
    word_rel = []

    # forward prop
    while to_do:
        curr = to_do.pop(0)
        # node is leaf
        if len(curr.kids) == 0:
            # activation function is the normalized tanh
            # compute hidden state
            curr.p = tanh(Wv.dot(curr.vec) + b)

        else:

            # - root isn't a part of this! 
            # - more specifically, the stanford dep. parser creates a superficial ROOT node
            #   associated with the word "root" that we don't want to consider during training
            # 'root' is the last one to be popped
            if len(to_do) == 0:
                # 'root' only has one kid, which is the root word
                ind, rel = curr.kids[0]
                curr.p = tree.get(ind).p

                continue

            # check if all kids are finished
            all_done = True
            for ind, rel in curr.kids:
                if tree.get(ind).finished == 0:
                    all_done = False
                    break

            # if not, push the node back onto the queue
            if not all_done:
                to_do.append(curr)
                continue

            # otherwise, compute p at node
            else:
                kid_sum = zeros( (d, 1) )
                for ind, rel in curr.kids:
                    if rel != 'root' and rel != 'ROOT':
                        curr_kid = tree.get(ind)
    
                        try:
                            curr_kid.v_rel = tanh(Wh.dot(curr_kid.p) + Wv.dot(curr.vec))
                            curr_kid.rel = rel_list.index(rel)
                            X_c.append(curr_kid.v_rel)
                            Xc_ind.append(ind)
                            
                            true_rel = zeros((len(rel_list), 1))
                            true_rel[curr_kid.rel] = 1.
                            curr_kid.true_rel = true_rel
                            Y.append(true_rel)
                                                        
                            kid_sum += rel_dict[rel].dot(curr_kid.v_rel)# / len(curr.kids)
                            # output relation and group
                            word_rel.append((curr.word, curr_kid.word, rel))
    
                        # - this shouldn't happen unless the parser spit out a seriously 
                        #   malformed tree
                        except KeyError:
                            print 'forward propagation error'
                            print tree.get_words()
                            print curr.word, rel, tree.get(ind).word
                
                curr.p = tanh(kid_sum + Wv.dot(curr.vec) + b)


        # error and delta
        if labels:
            true_label = zeros( (c, 1) )
            for i in range(c):
                if curr.trueLabel == i:
                    true_label[i] = 1
                    
            curr.true_class = true_label

        curr.finished = 1
        
    # compute autoencoders for relation prediction and parameter updates
    X_c = np.asarray(X_c).reshape((len(Xc_ind), d))
    Y = np.asarray(Y).reshape((len(Xc_ind), len(rel_list)))
    auto_loss = model.train(X_c, Y)
    grad_h = model.grad_h(X_c, Y)
    Y_pred, Y_class = model.predict(X_c)
    tree.error_aux = auto_loss
    ave_emb, group_id = model.get_emb(X_c)
    
    for ind, v, y in zip(Xc_ind, grad_h, Y_pred):
        tree.get(ind).grad_h = v
        tree.get(ind).predict_rel = y.reshape((len(rel_list), 1))
        tree.get(ind).rlabel_error = - (np.multiply(log(tree.get(ind).predict_rel), tree.get(ind).true_rel).sum())
    '''
    if rel_out:
        # output word, relation and group
        for gid, tpl, y_class in zip(group_id, word_rel, Y_class):
            rel_out.write(','.join(item for item in tpl))
            rel_out.write('--' + str(gid))
            rel_out.write('--' + rel_list[y_class])
            rel_out.write('\n')
    
    # evaluation on relation prediction
    match_rel = 0
    for y, y_class in zip(Y, Y_class):
        if y[y_class] == 1.:
            match_rel += 1
    return match_rel, Y.shape[0]        
    '''

        
# computes gradients for the given tree and increments existing gradients
# backpropagate on target domain
def backprop_t_auto(params, rel_list, tree, d, c, len_voc, grads, mixed = False):

    (rel_dict, Wv, Wh, b) = params
    # start with root's immediate kid (for same reason as forward prop)
    ind, rel = tree.get(0).kids[0]
    root = tree.get(ind)
    # operate on tuples of the form (node, parent delta)
    to_do = [ (root, zeros( (d, 1) ) ) ]

    while to_do:
        curr = to_do.pop()
        node = curr[0]
        #parent delta
        delta_down = curr[1]
        
        #delta_node
        curr_der = der_tanh(node.p)
        node.delta_full = np.multiply(delta_down, curr_der)
        # internal node
        if len(node.kids) > 0:
            num_kids = 0
            grads_Wv = 0
            grads_x = 0
            
            for ind, rel in node.kids:
                num_kids += 1
                curr_kid = tree.get(ind)
                
                #gradients on relation vector
                delta_rTov = (curr_kid.grad_h).reshape((d,1))
                delta_hTov = rel_dict[rel].T.dot(node.delta_full)# / len(node.kids)
                delta_v = np.multiply(delta_rTov + delta_hTov, der_tanh(curr_kid.v_rel))

                grads[0][rel] += node.delta_full.dot(curr_kid.v_rel.T)# / len(node.kids)
                grads_Wv += delta_v.dot(node.vec.T)
                grads[2] += delta_v.dot(curr_kid.p.T)
                grads_x += Wv.T.dot(delta_v).ravel()

                to_do.append( (curr_kid, Wh.T.dot(delta_v) ) )

            grads[1] += grads_Wv / num_kids
            if mixed:
                grads[4][50:, node.ind] += grads_x[50:] / num_kids + Wv.T.dot(node.delta_full).ravel()[50:]
            else:
                grads[4][:, node.ind] += grads_x / num_kids + Wv.T.dot(node.delta_full).ravel()
                
        #leaf node
        else:
            
            if mixed:
                grads[4][50:, node.ind] += Wv.T.dot(node.delta_full).ravel()[50:]
            else:
                grads[4][:, node.ind] += Wv.T.dot(node.delta_full).ravel()
        
                
        grads[1] += node.delta_full.dot(node.vec.T)        
        grads[3] += node.delta_full

# with autoencoder
# backpropagate on source domain
def backprop_s_auto(params, rel_list, tree, d, c, len_voc, grads, mixed = False):

    (rel_dict, Wv, Wh, b) = params
    # start with root's immediate kid (for same reason as forward prop)
    ind, rel = tree.get(0).kids[0]
    root = tree.get(ind)
    # operate on tuples of the form (node, parent delta)
    to_do = [ (root, zeros( (d, 1) ) ) ]

    while to_do:
        curr = to_do.pop()
        node = curr[0]
        #parent delta
        delta_down = curr[1]
        delta = node.grad_emb
        #delta_node
        curr_der = der_tanh(node.p)
        node.delta_full = np.multiply(delta_down + delta, curr_der)
        # internal node
        if len(node.kids) > 0:
            num_kids = 0
            grads_Wv = 0
            grads_x = 0
            
            for ind, rel in node.kids:
                num_kids += 1
                curr_kid = tree.get(ind)
                
                #gradients on relation vector
                delta_rTov = (curr_kid.grad_h).reshape((d,1))
                delta_hTov = rel_dict[rel].T.dot(node.delta_full)# / len(node.kids)
                delta_v = np.multiply(delta_rTov + delta_hTov, der_tanh(curr_kid.v_rel))

                grads[0][rel] += node.delta_full.dot(curr_kid.v_rel.T)# / len(node.kids)
                grads_Wv += delta_v.dot(node.vec.T)
                grads[2] += delta_v.dot(curr_kid.p.T)
                grads_x += Wv.T.dot(delta_v).ravel()

                to_do.append( (curr_kid, Wh.T.dot(delta_v) ) )

            grads[1] += grads_Wv / num_kids
            if mixed:
                grads[4][50:, node.ind] += grads_x[50:] / num_kids + Wv.T.dot(node.delta_full).ravel()[50:]
            else:
                grads[4][:, node.ind] += grads_x / num_kids + Wv.T.dot(node.delta_full).ravel()
                
        #leaf node
        else:
            
            if mixed:
                grads[4][50:, node.ind] += Wv.T.dot(node.delta_full).ravel()[50:]
            else:
                grads[4][:, node.ind] += Wv.T.dot(node.delta_full).ravel()
        
                
        grads[1] += node.delta_full.dot(node.vec.T)        
        grads[3] += node.delta_full
        
        

