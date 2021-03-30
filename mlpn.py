import numpy as np

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

def classifier_output(x, params):
    Z = [np.zeros(1)]
    V = [X]
    for l in range(1, self.L):
        Z_l = np.dot(self.W[l], V[l - 1]) + self.b[l].reshape(-1, 1)
        if l == 1:
            V_l = sigmoid(Z_l)
        else:
            V_l = relu(Z_l)
        V.append(V_l)
        Z.append(Z_l)
    Z_L = np.dot(self.W[self.L], V[self.L - 1]) + self.b[self.L].reshape(-1, 1)
    V_L = softmax(Z_L)

    V.append(V_L)
    Z.append(Z_L)

    return V, Z
    return probs

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
    # YOU CODE HERE
    return ...

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


