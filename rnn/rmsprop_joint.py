from numpy import *

class Rmsprop(): 

    def __init__(self, dim):
        self.dim = dim
        self.eps = 1e-6
        self.decay = 0.9
        #self.eps = 1e-3

        # initial learning rate
        #self.learning_rate = 0.05  #original 100-dim depnn
        #self.learning_rate = 0.0001  #crf 100-dim
        #self.learning_rate = 0.02
        self.learning_rate = 0.0001
        # stores sum of squared gradients 
        self.h = zeros(self.dim)

    def rescale_update(self, gradients):
        curr_params = self.decay * self.h + (1 - self.decay) * gradients ** 2
        updates = self.learning_rate * gradients / sqrt(curr_params + self.eps)
        self.params = curr_params

        return updates

    def reset_weights(self):
        self.h = zeros(self.dim)
