"""
GRU module
with batch and dropout
input taken from the output of recursive neural network
"""

import theano
import numpy
import os
import cPickle

from theano import tensor as T
# from collections import OrderedDict
from theano.compat.python2x import OrderedDict

dtype = theano.config.floatX
uniform = numpy.random.uniform
sigma = T.nnet.sigmoid
softmax = T.nnet.softmax
 
class model(object):

    def __init__(self, nh, nc, ne, de, cs, decay):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size 
        decay :: adaptive learning rate
        '''
        # parameters of the model
        # weights for GRU
        n_in = de * cs
        n_hidden = n_i = n_c = n_f = nh
        n_y = nc
        # forward pass
        self.W_xi = theano.shared(0.2 * uniform(-1.0, 1.0, (n_in, n_i)).astype(dtype))
        self.W_hi = theano.shared(0.2 * uniform(-1.0, 1.0, (n_hidden, n_i)).astype(dtype))
        self.b_i = theano.shared(numpy.cast[dtype](uniform(-0.5,.5,size = n_i)))
        self.W_xf = theano.shared(0.2 * uniform(-1.0, 1.0, (n_in, n_f)).astype(dtype))
        self.W_hf = theano.shared(0.2 * uniform(-1.0, 1.0, (n_hidden, n_f)).astype(dtype))
        self.b_f = theano.shared(numpy.cast[dtype](uniform(0, 1.,size = n_f)))
        self.W_xc = theano.shared(0.2 * uniform(-1.0, 1.0, (n_in, n_c)).astype(dtype))
        self.W_hc = theano.shared(0.2 * uniform(-1.0, 1.0, (n_hidden, n_c)).astype(dtype))
        self.b_c = theano.shared(numpy.zeros(n_c, dtype=dtype))

        self.c0 = theano.shared(numpy.zeros(n_hidden, dtype=dtype))
        self.h0 = T.tanh(self.c0)
        self.W_hy = theano.shared(0.2 * uniform(-1.0, 1.0, (n_hidden, n_y)).astype(dtype))
        self.b_y = theano.shared(numpy.zeros(n_y, dtype=dtype))
        '''
        # backward pass
        self.bW_xi = theano.shared(0.2 * uniform(-1.0, 1.0, (n_in, n_i)).astype(dtype))
        self.bW_hi = theano.shared(0.2 * uniform(-1.0, 1.0, (n_hidden, n_i)).astype(dtype))
        self.bb_i = theano.shared(numpy.cast[dtype](uniform(-0.5,.5,size = n_i)))
        self.bW_xf = theano.shared(0.2 * uniform(-1.0, 1.0, (n_in, n_f)).astype(dtype))
        self.bW_hf = theano.shared(0.2 * uniform(-1.0, 1.0, (n_hidden, n_f)).astype(dtype))
        self.bb_f = theano.shared(numpy.cast[dtype](uniform(0, 1.,size = n_f)))
        self.bW_xc = theano.shared(0.2 * uniform(-1.0, 1.0, (n_in, n_c)).astype(dtype))
        self.bW_hc = theano.shared(0.2 * uniform(-1.0, 1.0, (n_hidden, n_c)).astype(dtype))
        self.bb_c = theano.shared(numpy.zeros(n_c, dtype=dtype))

        self.bc0 = theano.shared(numpy.zeros(n_hidden, dtype=dtype))
        self.bh0 = T.tanh(self.bc0)
        self.bW_hy = theano.shared(0.2 * uniform(-1.0, 1.0, (n_hidden, n_y)).astype(dtype))
        '''
        
        
        # bundle weights
        self.params = [self.W_xi, self.W_hi, self.b_i, self.W_xf, self.W_hf, \
                       self.b_f, self.W_xc, self.W_hc, self.b_c, self.W_hy, self.b_y]
        
        self.names  = ['W_xi', 'W_hi', 'b_i', 'W_xf', 'W_hf', 'b_f', \
                       'W_xc', 'W_hc', 'b_c', 'W_xo', 'W_ho', 'b_o', 'W_hy', 'b_y', 'c0']
        # for dropout               
        self.allcache = [theano.shared(W.get_value() * numpy.asarray(0., dtype=dtype)) for W in self.params]
        
        # input context vectors in a batch
        embs = T.ftensor3('embs')
        mask = T.ivector('mask')
        idxs = T.itensor3()# as many columns as context window size/lines as words in the sentence
        x, _ = theano.scan(lambda idx, emb: emb[idx].reshape((idx.shape[0], de*cs)), sequences=[idxs, embs])
        y = T.imatrix('y') 
                

        def recurrence(x_t, h_tm1):
            i_t = sigma(theano.dot(x_t, self.W_xi) + theano.dot(h_tm1, self.W_hi) + self.b_i)
            f_t = sigma(theano.dot(x_t, self.W_xf) + theano.dot(h_tm1, self.W_hf) + self.b_f)
            c_t = T.tanh(theano.dot(x_t, self.W_xc) + theano.dot(h_tm1 * f_t, self.W_hc) + self.b_c)
            h_t = (T.ones_like(i_t) - i_t) * h_tm1 + i_t * c_t

            s_t = softmax(theano.dot(h_t, self.W_hy) + self.b_y)[0]
            
            return [h_t, s_t]

        '''    
        def brecurrence(x_t, feat_t, h_tm1, c_tm1):
            i_t = sigma(theano.dot(x_t, self.bW_xi) + theano.dot(h_tm1, self.bW_hi) + self.bb_i)
            f_t = sigma(theano.dot(x_t, self.bW_xf) + theano.dot(h_tm1, self.bW_hf) + self.bb_f)
            c_t = T.tanh(theano.dot(x_t, self.bW_xc) + theano.dot(h_tm1 * f_t, self.bW_hc) + self.bb_c)
            h_t = (T.ones_like(i_t) - i_t) * h_tm1 + i_t * c_t
            return [h_t, c_t]
        '''
        
        # loss for each sentence, m is mask
        def sent_model(x_sent, m, y_sent):        
            [h, s], _ = theano.scan(fn=recurrence, sequences=[x_sent], outputs_info=[self.h0, None])
            max_y, _ = theano.scan(lambda v, l: T.log(v)[l], sequences=[s[:m], y_sent[:m]])
            nll = -T.mean(max_y)
            return nll
        
        # prediction for each sentence, m is mask
        def pred_model(x_sent, m):        
            [h, s], _ = theano.scan(fn=recurrence, sequences=[x_sent], outputs_info=[self.h0, None])
            y_pred = T.argmax(s, axis=1)
            return y_pred
        
        nll_all, _ = theano.scan(fn=sent_model, sequences=[x, mask, y])
        nll_all = T.mean(nll_all)
        y_pred, _ = theano.scan(fn=pred_model, sequences=[x, mask])
        
        # cost and gradients and learning rate
        lr = T.scalar('lr')        
        gradients = T.grad( nll_all, self.params )
        
        # rmsprop
        allcache = [decay * cacheW + (1 - decay) * gradient ** 2 for cacheW, gradient in zip(self.allcache, gradients)]
        updates = OrderedDict([( p, p-lr*g/T.sqrt(cache+1e-6) ) for p, g, cache in zip( self.params , gradients, allcache)] \
                                + [(w, new_w) for w, new_w in zip(self.allcache, allcache)])
        # gradients for input context vectors
        emb_update = T.grad(nll_all, embs)
        
        # theano functions
        self.predict = theano.function(inputs=[idxs, embs, mask], outputs=y_pred, allow_input_downcast=True)
        self.train = theano.function(inputs=[idxs, embs, y, lr, mask], outputs=nll_all, updates=updates, allow_input_downcast=True)

        #self.normalize = theano.function(inputs=[], updates={self.emb: self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0,'x')})
        #self.update_emb = theano.function(inputs=[new, idxs], updates={self.emb[idxs]: theano.shared(new[idxs].get_value())})

        #add returning gradients for embedding
        self.grad = theano.function(inputs=[idxs,embs,y,mask],outputs=emb_update, allow_input_downcast=True)
        
        #self.hidden = theano.function(inputs=[idxs,emb], outputs=h, allow_input_downcast=True)

    def save(self, filename):   
        cPickle.dump([param.get_value() for param in self.params], filename)
    
    def load(self, filename):
        params = cPickle.load(open(filename, 'rb'))
        [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xc, W_hc, b_c, W_xo, W_ho, b_o, W_hy, b_y, c0] = params
        self.W_xi = W_xi
        self.W_hi = W_hi
        self.b_i = b_i
        self.W_xf = W_xf
        self.W_hf = W_hf
        self.b_f = b_f
        self.W_xc = W_xc
        self.W_hc = W_hc
        self.b_c = b_c
        self.W_xo = W_xo
        self.W_ho = W_ho
        self.b_o = b_o
        self.W_hy = W_hy
        self.b_y = b_y
        self.c0 = c0
        
            