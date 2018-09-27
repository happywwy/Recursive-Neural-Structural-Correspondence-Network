import numpy as np
import theano


def uniform(shape, scale=0.05, name=None):
    np.random.seed(5)
    return theano.shared(np.random.uniform(low=-scale, high=scale, size=shape).astype(theano.config.floatX), name=name)


def w2v(shape, name=None):
    np.random.seed(5)
    e = np.empty(shape, dtype=theano.config.floatX)
    for i in range(shape[0]):
        e[i] = (np.random.rand(shape[1]) - 0.5) / shape[1]
    return theano.shared(e, name=name)


def zeros(shape, name=None):
    return theano.shared(np.zeros(shape, dtype=theano.config.floatX), name=name)