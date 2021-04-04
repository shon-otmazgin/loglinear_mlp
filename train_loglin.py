import loglinear as ll
import random
import numpy as np
from utils import F2I, L2I

STUDENT = {'name': 'Royi Rassin',
           'ID': '311334734',
           'name2': 'Shon Otzmagin',
           'ID2': '305394975'
           }

def feats_to_vec(features):
    arr = np.zeros(len(F2I))
    for f in features:
        try:
            arr[F2I[f]] += 1
        except KeyError:
            # f not in vocab (top 600 bigram)
            continue
    return arr

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        x = feats_to_vec(features)
        y_pred = ll.predict(x, params)
        good += 1 if y_pred==L2I[label] else 0
        bad += 1 if y_pred!=L2I[label] else 0
    return good / (good + bad)

def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in range(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features) # convert features to a vector.
            y = L2I[label]                  # convert the label to number if needed.
            loss, grads = ll.loss_and_gradients(x,y,params)
            cum_loss += loss

            W, b = params
            gW, gb = grads
            W_new = W - (learning_rate*gW)
            b_new = b - (learning_rate * gb)
            params = [W_new, b_new]

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return params

if __name__ == '__main__':
    from utils import TRAIN as train_data
    from utils import DEV as dev_data
    from utils import TEST as test_data
    from utils import I2L

    num_iterations = 10
    learning_rate = 1e-3
    in_dim = len(F2I)
    out_dim = len(L2I)

    params = ll.create_classifier(in_dim, out_dim)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)

    preds = []
    for features in test_data:
        x = feats_to_vec(features)
        preds.append(ll.predict(x, trained_params))

    # with open('test.pred', 'w') as f:
    #     for y_hat in preds:
    #         f.write(f'{I2L[y_hat]}\n')

