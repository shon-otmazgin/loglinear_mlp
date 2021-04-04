import numpy as np

from loglinear import softmax
from mlp1 import tanh, d_tanh

STUDENT = {'name': 'Royi Rassin',
           'ID': '311334734',
           'name2': 'Shon Otzmagin',
           'ID2': '305394975'
           }
Z, V = [], []

def classifier_output(x, params):
    global Z, V
    x = np.array(x).reshape(-1, 1)

    Z = [np.zeros(1)]
    V = [x]
    L = int((len(params) / 2) - 1)
    for l in range(L):
        W, b = params[l*2], params[(l*2)+1]

        Z_hid = np.dot(W.T, V[l]) + b.reshape(-1, 1)  # [hid_dim, 1]
        V_hid = tanh(Z_hid)                             # [hid_dim, 1]

        Z.append(Z_hid)
        V.append(V_hid)

    W, b = params[L*2], params[(L*2) + 1]
    Z_out = np.dot(W.T, V[L]) + b.reshape(-1, 1)  # [out_dim, 1]
    V_out = softmax(Z_out)                        # [out_dim, 1]

    V.append(Z_out)
    Z.append(V_out)

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

    L = int((len(params) / 2) - 1)
    for l in range(L, -1, -1):
        Z_hid, V_hid = Z[l], V[l]

        gW = np.dot(G, V_hid.T).T  # [hid_dim, out_dim]
        gb = G

        grads.insert(0, gb)
        grads.insert(0, gW)

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
    from grad_check import gradient_check

    W, b = create_classifier([3, 4])

    def _loss_and_W_grad(W):
        global b
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b])
        return loss, grads[0]

    def _loss_and_b_grad(b):
        global W
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b])
        return loss, grads[1]

    for _ in range(10):
        W = np.random.randn(W.shape[0], W.shape[1])
        b = np.random.randn(b.shape[0])
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)

    W1, b1, W2, b2, W3, b3 = create_classifier([3, 20, 30, 4])

    def _loss_and_W1_grad(W1):
        global b1, W2, b2, W3, b3
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W1, b1, W2, b2, W3, b3])
        return loss, grads[0]

    def _loss_and_b1_grad(b1):
        global W1, W2, b2, W3, b3
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W1, b1, W2, b2, W3, b3])
        return loss, grads[1]

    def _loss_and_W2_grad(W2):
        global W1, b1, b2, W3, b3
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W1, b1, W2, b2, W3, b3])
        return loss, grads[2]

    def _loss_and_b2_grad(b2):
        global W1, b1, W2, W3, b3
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W1, b1, W2, b2, W3, b3])
        return loss, grads[3]

    def _loss_and_W3_grad(W3):
        global W1, b1, W2, b2, b3
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W1, b1, W2, b2, W3, b3])
        return loss, grads[4]

    def _loss_and_b3_grad(b3):
        global W1, b1, W2, b2, W3
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W1, b1, W2, b2, W3, b3])
        return loss, grads[5]

    for _ in range(10):
        W1 = np.random.randn(W1.shape[0], W1.shape[1])
        b1 = np.random.randn(b1.shape[0])
        W2 = np.random.randn(W2.shape[0], W2.shape[1])
        b2 = np.random.randn(b2.shape[0])
        W3 = np.random.randn(W3.shape[0], W3.shape[1])
        b3 = np.random.randn(b3.shape[0])

        gradient_check(_loss_and_b1_grad, b1)
        gradient_check(_loss_and_W1_grad, W1)

        gradient_check(_loss_and_b2_grad, b2)
        gradient_check(_loss_and_W2_grad, W2)

        gradient_check(_loss_and_b3_grad, b3)
        gradient_check(_loss_and_W3_grad, W3)



