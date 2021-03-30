import numpy as np

from loglinear import softmax
from mlp1 import tanh, d_tanh

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}
Z, V = [], []

def classifier_output(x, params):
    global Z, V
    x = np.array(x).reshape(-1, 1)

    Z = []
    V = [x]
    L = (len(params) / 2) - 1
    for l in range(L):
        W, b = params[l*2], params[(l*2)+1]

        Z_hid = np.dot(W.T, V[l-1]) + b.reshape(-1, 1)  # [hid_dim, 1]
        V_hid = tanh(Z_hid)                             # [hid_dim, 1]

        Z.append(Z_hid)
        V.append(V_hid)

    W, b = params[L*2], params[(L*2) + 1]
    Z_out = np.dot(W.T, V[L]) + b.reshape(-1, 1)  # [out_dim, 1]
    V_out = softmax(Z_out)                        # [out_dim, 1]

    V.append(V_L)
    Z.append(Z_L)

    return V_out

def predict(x, params):
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """
    global Z, V
    x = np.array(x).reshape(-1, 1)

    grads = []

    probs = classifier_output(x, params)
    loss = -np.log(probs[y])

    E = probs       # [out_dim, 1]
    E[y] -= 1
    G = E.copy()    # [out_dim, 1]

    L = (len(params) / 2) - 1
    for l in range(L, -1, -1):
        Z_hid, V_hid = Z[l-1], V[l-1]

        gW = np.dot(G, V_hid.T).T  # [hid_dim, out_dim]
        gb = G

        grads.append(gW)
        grads.append(gb)

        E = np.dot(params[l*2], G)  # [hid_dim, 1]
        if l > 0:
            df = d_tanh(Z_hid)
        G = df * E                  # element-wise [hid_dim, 1]

    return loss, grads


def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """

    params = []
    # Initialize random W_i and b_i
    for i in range(1, len(dims)):
        x = np.sqrt(6 / (dims[i] + dims[i - 1]))
        params.append(np.random.uniform(low=-x, high=x, size=(dims[i-1], dims[i])).astype(np.float64))
        params.append(np.random.uniform(low=-x, high=x, size=(dims[i],)).astype(np.float64))

    return params


if __name__ == '__main__':
    dims = [300, 20, 30, 40, 5]
    params = create_classifier(dims)
    print([x.shape for x in params])

    dims = [300, 5, 4]
    params = create_classifier(dims)
    print([x.shape for x in params])

    dims = [300, 4]
    params = create_classifier(dims)
    print([x.shape for x in params])


