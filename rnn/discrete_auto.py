
"""
reduce label noises with auto-encoders
generate relation groups
"""

import theano
import theano.tensor as TT
import numpy

from functions import losses, optimizers
import cPickle
from numpy import linalg as LA
uniform = numpy.random.uniform

dtype = theano.config.floatX

class Groups():
    def __init__(self, input_dim, emb_dim, n_groups, nc, params=None):
        """
        :param input_dim: vocabulary size
        :param emb_dim: embedding dimension
        :param n_groups: number of relation groups
        :param W_g: relation group embedding
        :param W_enc: encoding matrix
        :param W_dec: decoding matrix
        :param W_y: classification matrix
        """
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.n_groups = n_groups
        self.nc = nc

        if params is not None:
            [W_g, W_enc, W_dec, b_dec, W_y, b_y] = params
            self.W_g = theano.shared(W_g)
            self.W_enc = theano.shared(W_enc)
            self.W_dec = theano.shared(W_dec)
            self.b_dec = theano.shared(b_dec)
            self.W_y = theano.shared(W_y)
            self.b_y = theano.shared(b_y)
        else:
            self.W_g = theano.shared(0.2 * uniform(-1.0, 1.0, (n_groups, emb_dim)).astype(dtype))        
            self.W_enc = theano.shared(0.2 * uniform(-1.0, 1.0, (input_dim, emb_dim)).astype(dtype))
            self.W_dec = theano.shared(0.2 * uniform(-1.0, 1.0, (emb_dim, input_dim)).astype(dtype))
            self.b_dec = theano.shared(numpy.zeros(input_dim, dtype=dtype))
            self.W_y = theano.shared(0.2 * uniform(-1.0, 1.0, (emb_dim, nc)).astype(dtype))
            self.b_y = theano.shared(numpy.zeros(nc, dtype=dtype))
            
            
        self.params = [self.W_g, self.W_enc, self.W_dec, self.b_dec, self.W_y, self.b_y]
        
    def build(self, loss1="crossentropy", loss2="mse", optimizer="rmsprop", lr=0.01, rho=0.9, epsilon=1e-6):
        self.loss1 = losses.get(loss1)
        self.loss2 = losses.get(loss2)
        optim = optimizers.get(optimizer, inst=False)

        if optim.__name__ == "RMSprop":
            self.optimizer = optim(lr=lr, rho=rho, epsilon=epsilon)
        elif optim.__name__ == "Adagrad":
            self.optimizer = optim(lr=lr, epsilon=epsilon)
        else:
            self.optimizer = optim(lr=lr)

        # get input to model
        self.X_c = TT.fmatrix(name="X_c")  # n_batches*input_dim
        # output label
        self.Y = TT.matrix(dtype=theano.config.floatX, name="Y")  # n_batches*nc
        self.X_recon, self.Y_pred, self.Y_class, self.ave_emb, self.group_ids = self.get_output()  # Y_pred: n_batches*nc

        # prediction_loss + reconstruction_loss + reg_loss
        train_loss_pred = self.get_loss1(self.Y, self.Y_pred) + self.get_loss2(self.X_c, self.X_recon)
        reg1_loss = TT.sqr(self.W_g).mean() + TT.sqr(self.W_enc).mean() + TT.sqr(self.W_dec).mean()
        Wg_norm, _ = theano.scan(lambda row: row / LA.norm(row), sequences=[self.W_g])
        inter_Wg = TT.dot(Wg_norm, TT.transpose(Wg_norm))
        reg2_loss = self.get_loss2(inter_Wg, TT.identity_like(inter_Wg))
        train_loss = train_loss_pred + 0.1 * reg1_loss + 0.0001 * reg2_loss

        updates = self.optimizer.get_updates(self.params, cost=train_loss)
        
        self.grad_h = theano.function(inputs=[self.X_c, self.Y],
                                      on_unused_input='warn',
                                      outputs=optimizers.get_gradients(train_loss, self.X_c),
                                      allow_input_downcast=True)
        self.train = theano.function(inputs=[self.X_c, self.Y], 
                                     on_unused_input='warn',
                                     outputs=train_loss_pred,
                                     updates=updates,
                                     allow_input_downcast=True,  
                                     mode=None
                                     )
        self.predict = theano.function(inputs=[self.X_c],
                                       on_unused_input='warn',
                                       outputs=[self.Y_pred, self.Y_class],
                                       allow_input_downcast=True
                                       )
                                      
        self.get_emb = theano.function(inputs=[self.X_c], 
                                     on_unused_input='warn',
                                     outputs=[self.ave_emb, self.group_ids],
                                     allow_input_downcast=True,  
                                     mode=None
                                     )
        
    
    # produce output from auto-encoder, including reconstructed embedding,
    # class prediction, etc. refer to the paper
    def get_output(self):
        # obtain context embeddings
        context_vec = self.X_c
        group_ave, group_ids = self.predict_group(self.W_g, self.W_enc, context_vec)  # n_batches
        # group specific embs only
        #W_group = W_p[TT.arange(W_p.shape[0]), group_ids]  # n_batches*emb_dim
        # reconstruct context vector from group_ave
        W_batch = TT.dot(group_ave, self.W_dec) + self.b_dec
        X_recon = W_batch
        # predict observed relations n_batches * nc
        y_pred, _ = theano.scan(lambda group_v, W, b: TT.nnet.softmax(TT.dot(group_v, W) + b)[0], \
                                sequences=[group_ave], non_sequences=[self.W_y, self.b_y])
        y_class = TT.argmax(y_pred, axis=1)
        return X_recon, y_pred, y_class, group_ave, group_ids
    
    # return unsupervised group prediction and average group embedding from auto-encoder
    def predict_group(self, W, Wenc, C):
        """
        :param W: n_groups * emb_dim
        :param C: context vectors; n_batches * input_dim
        :param Wenc: input_dim * emb_dim
        """
        W_inter = TT.dot(C, Wenc)
        #n_batches * n_groups
        group_score = TT.dot(W_inter, TT.transpose(W))
        group_prob, _ = theano.scan(lambda s: TT.nnet.softmax(s)[0], sequences=[group_score]) #n_batches * n_groups
        group_ave, _ = theano.scan(lambda v, m: TT.dot(v, m), sequences=[group_prob], non_sequences=[W])
        group_pred = group_prob.argmax(axis=1) 
        
        return group_ave, group_pred  

    def get_loss1(self, Y, Y_pred):
        return self.loss1(Y, Y_pred)
        
    def get_loss2(self, Y, Y_pred):
        return self.loss2(Y, Y_pred)
    
    def save(self, filename):
        
        cPickle.dump([param.get_value() for param in self.params], filename)


