import numpy as np

from loglinear import softmax

STUDENT = {'name': 'Royi Rassin, Shon Otzmagin',
           'ID': '311334734, 305394975'
           }

tanh = lambda x: np.tanh(x)
d_tanh = lambda x: 1.0 - np.tanh(x)**2

def classifier_output(x, params):
    x = np.array(x).reshape(-1, 1)

    W, b, U, b_tag = params

    Z_hid = np.dot(W.T, x) + b.reshape(-1, 1) # [hid_dim, 1]
    V_hid = tanh(Z_hid)                       # [hid_dim, 1]

    Z_out = np.dot(U.T, V_hid) + b_tag.reshape(-1, 1) # [out_dim, 1]
    V_out = softmax(Z_out)                            # [out_dim, 1]

    probs = V_out
    return probs

def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """
    x = np.array(x).reshape(-1, 1)

    W, b, U, b_tag = params

    probs = classifier_output(x, params)
    loss = -np.log(probs[y])

    Z_hid = np.dot(W.T, x) + b.reshape(-1, 1)          # [hid_dim, 1]
    V_hid = tanh(Z_hid)                                # [hid_dim, 1]

    # Z_out = np.dot(U.T, V_hid) + b_tag.reshape(-1, 1)  # [out_dim, 1]
    # V_out = softmax(Z_out)                             # [out_dim, 1]

    E = probs                                          # [out_dim, 1]
    E[y] -= 1
    G = E.copy()                                       # [out_dim, 1]

    gU = np.dot(G, V_hid.T).T                          #[hid_dim, out_dim]
    gb_tag = G

    E = np.dot(U, G)                                   # [hid_dim, 1]
    G = d_tanh(Z_hid) * E                              # element-wise [hid_dim, 1]

    gW = np.dot(G, x.T).T                              # [in_dim, hid_dim]
    gb = G

    return loss,[gW, gb, gU, gb_tag]

def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    x = np.sqrt(6 / (in_dim + hid_dim))

    W = np.random.uniform(low=-x, high=x, size=(in_dim, hid_dim)).astype(np.float64)
    b = np.random.uniform(low=-x, high=x, size=(hid_dim, )).astype(np.float64)

    x = np.sqrt(6 / (hid_dim + out_dim))
    U = np.random.uniform(low=-x, high=x, size=(hid_dim, out_dim)).astype(np.float64)
    b_tag = np.random.uniform(low=-x, high=x, size=(out_dim,)).astype(np.float64)

    return [W, b, U, b_tag]

if __name__ == '__main__':
    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    W, b, U, b_tag = create_classifier(3,5,4)

    def _loss_and_W_grad(W):
        global b, U, b_tag
        loss,grads = loss_and_gradients([1,2,3],0,[W, b, U, b_tag])
        return loss,grads[0]

    def _loss_and_b_grad(b):
        global W, U, b_tag
        loss,grads = loss_and_gradients([1,2,3],0,[W, b, U, b_tag])
        return loss,grads[1]

    def _loss_and_U_grad(U):
        global W, b, b_tag
        loss,grads = loss_and_gradients([1,2,3],0,[W, b, U, b_tag])
        return loss,grads[2]

    def _loss_and_b_tag_grad(b_tag):
        global W, b, U
        loss,grads = loss_and_gradients([1,2,3],0,[W, b, U, b_tag])
        return loss,grads[3]

    for _ in range(10):
        W = np.random.randn(W.shape[0],W.shape[1])
        b = np.random.randn(b.shape[0])
        U = np.random.randn(U.shape[0],U.shape[1])
        b_tag = np.random.randn(b_tag.shape[0])
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)
        gradient_check(_loss_and_b_tag_grad, b_tag)
        gradient_check(_loss_and_U_grad, U)

