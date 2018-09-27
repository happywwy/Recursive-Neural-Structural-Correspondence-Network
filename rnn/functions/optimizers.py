import numpy as np
import theano
import theano.tensor as T


def get_gradients(cost, params):
    loss_wrt_params = T.grad(cost, params)  # g_W_w, g_W_c etc.

    return loss_wrt_params


class SGD():
    # included T.inc_subtensor to compute grad only wrt to relevant: get_updates_subtens()
    # see get_updates for the original version
    def __init__(self, lr=0.01, momentum=0.9, decay=0.1, *args, **kwargs):
        self.__dict__.update(kwargs)
        self.__dict__.update(locals())
        self.iterations = theano.shared(np.cast[theano.config.floatX](0))
        #self.iterations = theano.shared(0)

    def get_updates_subtens(self, params, subparams, cost):
        grads = get_gradients(cost, subparams)
        updates = []
        assert len(params) == len(subparams) == len(grads)
        for p, p_sub, g in zip(params, subparams, grads):  # (W_w, g_W_w), (W_p, g_W_p)
            updates.append((p, T.inc_subtensor(p_sub, -self.lr * g)))

        return updates

    def _get_updates_subtens(self, params, subparams, cost):
        grads = get_gradients(cost, subparams)
        updates = []
        for p, p_sub, g, in zip(params, subparams, grads):
            #printed_var = theano.printing.Print("g")(g)
            # these are only the nonzero rows of the gradient matrix
            #g, i = g.owner.inputs[1:]
            # define the gradient descent update only for the nonzero gradient rows
            updates.append((p, T.inc_subtensor(p_sub, -self.lr * g)))
            #updates.append((p, p - self.lr * g))
        return updates, g#, printed_var

    def get_updates(self, params, cost):
        grads = get_gradients(cost, params)
        lr = self.lr * (1 / (1 + self.decay * self.iterations))
        updates = [(self.iterations, self.iterations+1.)]

        for p, g in zip(params, grads):  # (W_w, g_W_w), (W_p, g_W_p)
            m = theano.shared(np.asarray(np.zeros(p.get_value().shape), dtype=theano.config.floatX))
            v = self.momentum * m - lr * g # velocity
            updates.append((m, v))
            updates.append((p, p + v))

        return updates


class SGDDynamic():
    # reduce learning rate linearly

    def __init__(self, lr=0.025):
        self.lr = lr

    def get_updates_subtens(self, params, subparams, cost, new_lr):
        grads = get_gradients(cost, subparams)
        updates = []
        assert len(params) == len(subparams) == len(grads)
        for p, p_sub, g in zip(params, subparams, grads):  # (W_w, g_W_w), (W_p, g_W_p)
            updates.append((p, T.inc_subtensor(p_sub, -new_lr * g)))

        return updates

    def get_updates(self, params, cost):
        grads = get_gradients(cost, params)
        updates = []

        for p, g in zip(params, grads):  # (W_w, g_W_w), (W_p, g_W_p)
            updates.append((p, p - self.lr * g))

        return updates


class RMSprop():
    """
    Divide the learning rate for a weight by a running average of the magnitudes of recent gradients for that weight.
    """
    def __init__(self, lr=0.01, rho=0.9, epsilon=1e-6, *args, **kwargs):
        self.__dict__.update(kwargs)
        self.__dict__.update(locals())

    def get_updates(self, params, cost):
        grads = get_gradients(cost, params)
        accumulators = [theano.shared(np.asarray(np.zeros(p.get_value().shape), dtype=theano.config.floatX)) for p in params]
        updates = []

        for p, g, a in zip(params, grads, accumulators):
            new_a = self.rho * a + (1 - self.rho) * g ** 2  # update accumulator; moving average
            updates.append((a, new_a))

            new_p = p - self.lr * g / T.sqrt(new_a + self.epsilon)
            updates.append((p, new_p))

        return updates

    def get_updates_subtens(self, params, subparams, subinds, cost):
        """
        Based on https://groups.google.com/forum/#!msg/theano-users/Eu7kQFo2gik/o_oZl8-lsNYJ

        """
        grads = get_gradients(cost, subparams)
        accumulators = [theano.shared(np.asarray(np.zeros(p.get_value().shape), dtype=theano.config.floatX)) for p in params]
        assert len(subinds) == len(accumulators)
        subaccumulators = [a[subind] for subind, a in zip(subinds, accumulators)]

        updates = []
        assert len(params) == len(grads) == len(accumulators)
        for p, p_sub, g, a, a_sub in zip(params, subparams, grads, accumulators, subaccumulators):
            #printed_var = theano.printing.Print("g")(g)
            #new_a = T.inc_subtensor(a_sub, -self.rho * a_sub - ((1 - self.rho) * g ** 2) + a_sub)  # update accumulator
            upd_a = self.rho * a_sub - (1 - self.rho) * g ** 2
            new_a = T.set_subtensor(a_sub, upd_a)  # update accumulator
            updates.append((a, new_a))

            new_p = T.inc_subtensor(p_sub, -self.lr * g / T.sqrt(upd_a + self.epsilon))
            updates.append((p, new_p))

        return updates#,printed_var


class Adagrad():
    def __init__(self, lr=0.01, epsilon=1e-6, *args, **kwargs):
        self.__dict__.update(kwargs)
        self.__dict__.update(locals())

    def get_updates(self, params, cost):
        grads = get_gradients(cost, params)
        accumulators = [theano.shared(np.asarray(np.zeros(p.get_value().shape), dtype=theano.config.floatX)) for p in params]
        updates = []

        for p, g, a in zip(params, grads, accumulators):
            new_a = a + g ** 2  # update accumulator
            updates.append((a, new_a))

            new_p = p - self.lr * g / T.sqrt(new_a + self.epsilon)
            updates.append((p, new_p))
        return updates


def get(identifier, inst=True):
    return get_from_module(identifier, globals(), 'optimizer', instantiate=inst)

def get_from_module(identifier, module_params, module_name, instantiate=False):
    """
    Taken from keras.
    """
    if type(identifier) is str:
        res = module_params.get(identifier)
        if not res:
            raise Exception("Invalid {}: {}".format(module_name, identifier))
        if instantiate:
            return res()
        else:
            return res
    return identifier

# aliases
sgd = SGD
rmsprop = RMSprop