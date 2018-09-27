import theano.tensor as T
epsilon = 1.0e-7  # for float32!, could be smaller for f64


def mse(y, y_pred):
    """
    :param mask: vector/matrix/tensor of 1 and 0, same dim. as y_pred/y
    """
    return T.sqr(y_pred - y).mean()


def mse_masked(y, y_pred, mask):
    """
    :param mask: vector/matrix/tensor of 1 and 0, same dim. as y_pred/y
    """
    return T.sum(T.sqr(y_pred - y)*mask) / T.sum(mask)


def crossentropy(y_true, y_pred):
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    return T.nnet.binary_crossentropy(y_pred, y_true).mean()


def crossentropy_masked(y_true, y_pred, mask):
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    return T.sum(T.nnet.binary_crossentropy(y_pred, y_true) * mask) / T.sum(mask)


def get(identifier):
    return get_from_module(identifier, globals(), 'loss')

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